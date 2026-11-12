"""Suggested replies and summaries from an OpenAI-compatible chat API."""
import datetime
import threading
import time
from enum import Enum

import openai

from . import constants, conversation, prompts
from .db import AppDB as appdb
from .db import llm_responses as llmrdb
from .db import summaries as s
from tsutils import app_logging as al
from tsutils import duration


logger = al.get_module_logger(al.GPT_RESPONDER_LOGGER)

# Wait this long for the transcript to stop changing before asking for a suggestion,
# so one spoken sentence that arrives in two updates is answered once ...
RESPONSE_SETTLE_SECONDS = 0.6
# ... but never wait longer than this while someone keeps talking.
RESPONSE_SETTLE_LIMIT_SECONDS = 3.0

SUPPORTED_REASONING_EFFORTS = {
    'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'
}

PLACEHOLDER_API_KEYS = {'', 'API_KEY', 'YOUR_API_KEY', 'sk-...'}

STATUS_NO_KEY = 'no_key'        # no API key in the settings: suggestions are off
STATUS_READY = 'ready'
STATUS_WORKING = 'working'      # a request is running
STATUS_ERROR = 'error'          # the last request failed; last_error says why

NO_KEY_MESSAGE = 'No API key yet. Open Settings and enter the key of your AI provider.'
NOTHING_TO_ANSWER_MESSAGE = 'Nothing has been transcribed yet.'


class ResponderError(Exception):
    """A request to the AI provider failed; the message can be shown to the user as it is."""


class InferenceEnum(Enum):
    """Supported Chat Inference Providers
    """
    OPENAI = 1
    TOGETHER = 2


def has_api_key(api_key) -> bool:
    """True when the settings hold something that can be an API key (not the placeholder)."""
    return str(api_key or '').strip() not in PLACEHOLDER_API_KEYS


def describe_error(exception: Exception, model: str = '') -> str:
    """One sentence a user can act on, for an error of the AI provider."""
    if isinstance(exception, ResponderError):
        return str(exception)
    if isinstance(exception, openai.AuthenticationError):
        return 'The AI provider rejected the API key. Check the key in Settings.'
    if isinstance(exception, openai.PermissionDeniedError):
        return ('The AI provider refused the request (403). The key may not have access to this model, '
                'or the service is not available in your region: a VPN or another provider (Base URL in '
                'Settings) can help.')
    if isinstance(exception, openai.RateLimitError):
        if 'quota' in str(exception).lower() or 'billing' in str(exception).lower():
            return 'The account of this API key has no credit left. Check billing at your AI provider.'
        return 'The AI provider is limiting requests (too many at once). It is tried again automatically.'
    if isinstance(exception, openai.NotFoundError):
        return f'The AI provider does not know the model "{model}". Check Model and Base URL in Settings.'
    if isinstance(exception, openai.APITimeoutError):
        return 'The AI provider did not answer in time. Check the internet connection (or VPN).'
    if isinstance(exception, openai.APIConnectionError):
        return 'The AI provider could not be reached. Check the internet connection, VPN and the Base URL in Settings.'
    if isinstance(exception, openai.BadRequestError):
        return f'The AI provider did not accept the request: {_short(exception)}'
    if isinstance(exception, openai.APIStatusError):
        return f'The AI provider reported an error ({exception.status_code}): {_short(exception)}'
    return f'The suggestion failed: {_short(exception)}'


def _short(exception: Exception, limit: int = 220) -> str:
    body = getattr(exception, 'body', None)
    text = body.get('message') if isinstance(body, dict) and body.get('message') else str(exception)
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + '…'


def test_connection(api_key: str, base_url: str, model: str, timeout: float = 20.0):
    """Send one very small request with these settings. Returns (worked, message for the user)."""
    if not has_api_key(api_key):
        return False, 'Enter an API key first.'
    if not str(model or '').strip():
        return False, 'Enter a model name first.'
    client = None
    try:
        client = openai.OpenAI(api_key=api_key.strip(), base_url=(base_url or '').strip() or None,
                               timeout=timeout, max_retries=0)
        client.chat.completions.create(
            model=model.strip(),
            messages=[{'role': 'user', 'content': 'Reply with the single word: ok'}],
        )
        return True, f'It works: "{model.strip()}" answered.'
    except Exception as exception:  # pylint: disable=broad-except
        return False, describe_error(exception, model)
    finally:
        if client is not None:
            client.close()


class GPTResponder:
    """Handles all interactions with openAI LLM / ChatGPT
    """
    # By default we do not ping LLM to get data
    enabled: bool = False
    model: str = None
    llm_client = None
    settings_section: str = 'OpenAI'

    def __init__(self,
                 config: dict,
                 convo: conversation.Conversation,
                 file_name: str,
                 save_to_file: bool = False,
                 openai_module=openai):
        logger.info(GPTResponder.__name__)
        # This var is used by UI to populate the response textbox
        self.response = prompts.INITIAL_RESPONSE
        self.llm_response_interval = config['General']['llm_response_interval']
        self.conversation = convo
        self.config = config
        self.save_response_to_file = save_to_file
        self.response_file = file_name
        self.openai_module = openai_module
        self.status = STATUS_READY
        self.last_error: str | None = None
        self._request_lock = threading.Lock()       # one reply request at a time
        self._client_lock = threading.Lock()
        self.configure()

    # ----------------------------------------------------------------------------------------
    # Settings
    # ----------------------------------------------------------------------------------------
    def _get_settings_section(self, provider: str) -> str:
        """Get the settings section based on the chat inference provider."""
        if provider == 'openai':
            return 'OpenAI'
        elif provider == 'together':
            return 'Together'
        raise ValueError(f"Unsupported chat inference provider: {provider}")

    def _get_api_settings(self, settings_section: str):
        """Retrieve API settings from the configuration."""
        api_key = self.config[settings_section]['api_key']
        base_url = self.config[settings_section]['base_url']
        model = self.config[settings_section]['ai_model']
        return api_key, base_url, model

    def configure(self):
        """(Re)create the client from the settings as they are now.

        Called at start-up and after the settings window saves, so a new key, model or
        Base URL is used from the next request on without restarting the program.
        """
        api_key, base_url, model = self._get_api_settings(self.settings_section)
        api_key = str(api_key or '').strip()
        with self._client_lock:
            old_client = self.llm_client
            self.model = str(model or '').strip()
            if has_api_key(api_key):
                # The client's own retries are off: a retry of a live suggestion is out of date,
                # and the next change of the transcript asks again anyway.
                self.llm_client = self.openai_module.OpenAI(
                    api_key=api_key, base_url=str(base_url or '').strip() or None, max_retries=0)
                self._set_status(STATUS_READY)
            else:
                self.llm_client = None
                self._set_status(STATUS_NO_KEY, NO_KEY_MESSAGE)
        if old_client is not None:
            try:
                old_client.close()
            except Exception:  # pylint: disable=broad-except
                pass

    def create_client(self, api_key: str, base_url: str = None):
        """Create and initialize an OpenAI API compatible client."""
        if not api_key:
            raise ValueError("API key is required")
        try:
            if self.llm_client is not None:
                self.llm_client.close()
            self.llm_client = self.openai_module.OpenAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            raise ConnectionError(f"Failed to create OpenAI client: {e}")

    def _set_status(self, status: str, error: str | None = None):
        self.status = status
        self.last_error = error

    def _client(self):
        client = self.llm_client
        if client is None:
            raise ResponderError(NO_KEY_MESSAGE)
        return client

    def _get_openai_settings(self, settings_section: str = 'OpenAI'):
        """Retrieve request settings without inventing unsupported parameters."""
        timeout = self.config['OpenAI']['response_request_timeout_seconds']
        provider_config = self.config[settings_section]
        temperature = provider_config.get(
            'temperature', self.config['OpenAI'].get('temperature')
        )
        reasoning_effort = provider_config.get('reasoning_effort')
        if reasoning_effort not in (None, ''):
            if not isinstance(reasoning_effort, str):
                raise ValueError('reasoning_effort must be a string or null')
            reasoning_effort = reasoning_effort.lower()
            if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
                allowed = ', '.join(sorted(SUPPORTED_REASONING_EFFORTS))
                raise ValueError(f'Unsupported reasoning_effort. Expected one of: {allowed}')
        else:
            reasoning_effort = None
        return timeout, temperature, reasoning_effort

    def _build_chat_completion_request(self, messages, timeout,
                                       temperature=None,
                                       reasoning_effort=None) -> dict:
        """Build a streaming request, omitting optional unset parameters."""
        request = {
            'model': self.model,
            'messages': messages,
            'timeout': timeout,
            'stream': True,
        }
        if temperature is not None:
            request['temperature'] = temperature
        if reasoning_effort is not None:
            request['reasoning_effort'] = reasoning_effort
        return request

    # ----------------------------------------------------------------------------------------
    # Requests
    # ----------------------------------------------------------------------------------------
    @staticmethod
    def _stream_text(stream, on_text=None) -> str:
        """Collect a streamed answer; ``on_text`` gets the text so far after each piece."""
        collected = ''
        for chunk in stream:
            if not chunk.choices:           # e.g. a final usage chunk
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                collected += piece
                if on_text is not None:
                    on_text(collected)
        return collected

    def _run(self, action):
        """Run one request; keep the status and last error up to date. Returns None on failure."""
        self._set_status(STATUS_WORKING)
        try:
            result = action()
        except Exception as exception:  # pylint: disable=broad-except
            message = describe_error(exception, self.model)
            logger.error(f'LLM request failed: {exception!r}')
            print(f'[ERROR] {message}')
            self._set_status(STATUS_NO_KEY if self.llm_client is None else STATUS_ERROR, message)
            return None
        self._set_status(STATUS_READY)
        return result

    def summarize(self) -> str:
        """Ask the LLM for a summary of the conversation. Raises ResponderError with the reason
        when that is not possible."""
        logger.info(GPTResponder.summarize.__name__)

        def action():
            client = self._client()
            if not self.conversation.has_spoken_rows():
                raise ResponderError(NOTHING_TO_ANSWER_MESSAGE)
            with duration.Duration(name='OpenAI Summarize', screen=False):
                timeout: int = self.config['OpenAI']['summarize_request_timeout_seconds']
                _, temperature, reasoning_effort = self._get_openai_settings(self.settings_section)
                prompt_content = self.conversation.get_merged_conversation_summary()
                request = self._build_chat_completion_request(
                    prompts.create_multiturn_prompt(prompt_content), timeout, temperature, reasoning_effort)
                summary = self._stream_text(client.chat.completions.create(**request))
            last_convo_id = self._last_convo_id(prompt_content)
            if last_convo_id is not None:
                summary_obj = appdb().get_object(s.TABLE_NAME)
                summary_obj.insert_summary(appdb().get_invocation_id(), last_convo_id, summary)
            return summary

        summary = self._run(action)
        if summary is None:
            raise ResponderError(self.last_error or 'The summary could not be created.')
        return summary

    @staticmethod
    def _last_convo_id(rows: list):
        """Database id of the newest stored row among these conversation rows."""
        for row in reversed(rows):
            if row[2] is not None and int(row[2]) >= 0:
                return int(row[2])
        return None

    def _get_llm_response(self, messages, temperature, timeout,
                          reasoning_effort=None) -> str:
        """Send a request to the LLM and process the streaming response."""
        with duration.Duration(name='OpenAI Chat Completion', screen=False):
            request = self._build_chat_completion_request(
                messages, timeout, temperature, reasoning_effort
            )
            stream = self._client().chat.completions.create(**request)
            return self._stream_text(
                stream,
                lambda text: self._update_conversation(persona=constants.PERSONA_ASSISTANT,
                                                       response=text, update_previous=True))

    def _insert_response_in_db(self, last_convo_id: int, response: str):
        """Insert the generated response into the database."""
        inv_id = appdb().get_invocation_id()
        llmr_obj: llmrdb.LLMResponses = appdb().get_object(llmrdb.TABLE_NAME)
        llmr_obj.insert_response(inv_id, last_convo_id, response)

    def generate_response_from_transcript_no_check(self) -> str:
        """Ask the LLM for a suggested reply to the conversation now (also when automatic
        suggestions are off). Returns the reply, or None when it failed (see ``last_error``)."""
        logger.info(GPTResponder.generate_response_from_transcript_no_check.__name__)

        def action():
            self._client()
            if not self.conversation.has_spoken_rows():
                raise ResponderError(NOTHING_TO_ANSWER_MESSAGE)
            timeout, temperature, reasoning_effort = self._get_openai_settings(self.settings_section)
            rows = self.conversation.get_merged_conversation_response(
                length=constants.MAX_TRANSCRIPTION_PHRASES_FOR_LLM)
            text = self._get_llm_response(
                prompts.create_multiturn_prompt(rows), temperature, timeout, reasoning_effort)
            last_convo_id = self._last_convo_id(rows)
            if text and last_convo_id is not None:
                self._insert_response_in_db(last_convo_id, text)
            return text

        with self._request_lock:
            response = self._run(action)
        if response:
            self._save_response_to_file(response)
        return response

    def warm_up(self):
        """Open the connection to the provider in the background at start-up, so the first
        suggestion does not also wait for DNS and the TLS handshake. Costs nothing."""
        client = self.llm_client
        if client is None:
            return
        try:
            client.models.list(timeout=10)
        except Exception as exception:  # pylint: disable=broad-except
            # Not every compatible provider lists models; a real problem shows at the first request.
            logger.info(f'Warm-up request did not complete: {exception!r}')

    def process_response(self, input_str: str) -> str:
        """Remove the parts of an LLM answer that are not the reply itself."""
        if input_str is None:
            raise ValueError("input_str cannot be None")
        return prompts.strip_brackets(input_str)

    def generate_response_from_transcript(self) -> str:
        """Ask for a suggestion if automatic suggestions are on."""
        logger.info("generate_response_from_transcript called")

        if not self.enabled:
            return ''

        return self.generate_response_from_transcript_no_check()

    def generate_response_for_selected_text(self, text: str):
        """Ask the LLM for a reply to the text the user selected in the transcript."""
        logger.info(GPTResponder.generate_response_for_selected_text.__name__)

        def action():
            client = self._client()
            messages = prompts.create_prompt_for_text(text=text, config=self.config)
            if messages is None:
                raise ResponderError('Select some text in the transcript first.')
            timeout, temperature, reasoning_effort = self._get_openai_settings(self.settings_section)
            with duration.Duration(name='OpenAI Chat Completion Selected', screen=False):
                request = self._build_chat_completion_request(messages, timeout, temperature, reasoning_effort)
                stream = client.chat.completions.create(**request)
                return self._stream_text(
                    stream,
                    lambda so_far: self._update_conversation(persona=constants.PERSONA_ASSISTANT,
                                                             response=so_far, update_previous=True))

        with self._request_lock:
            response = self._run(action)
        if response:
            self._save_response_to_file(response)
        return response

    def _save_response_to_file(self, text: str):
        if self.save_response_to_file:
            try:
                with open(file=self.response_file, mode="a", encoding='utf-8') as f:
                    f.write(f'{datetime.datetime.now()} - {text}\n')
            except OSError as exception:
                logger.error(f'Could not write {self.response_file}: {exception}')

    def _update_conversation(self, response, persona, update_previous=False):
        """Update the internaal conversation state"""
        if response != '':
            self.response = response
            self.conversation.update_conversation(persona=persona,
                                                  text=response,
                                                  time_spoken=datetime.datetime.utcnow(),
                                                  update_previous=update_previous)

    def respond_to_transcriber(self, transcriber):
        """Thread method: ask the LLM for a new suggestion whenever the transcript changes.

        It wakes up on the transcriber's change event instead of polling every
        ``llm_response_interval`` seconds (which added up to a whole interval of delay before
        a question was even sent). Updates that arrive close together are answered once, and
        two requests are still never started less than ``llm_response_interval`` apart.
        """
        changed = transcriber.transcript_changed_event
        last_request = None
        while True:
            changed.wait()
            settle_until = time.monotonic() + RESPONSE_SETTLE_LIMIT_SECONDS
            while True:
                changed.clear()
                remaining = settle_until - time.monotonic()
                if remaining <= 0 or not changed.wait(min(RESPONSE_SETTLE_SECONDS, remaining)):
                    break
            if not self.enabled or self.llm_client is None:
                continue
            if last_request is not None:
                wait = self.llm_response_interval - (time.monotonic() - last_request)
                if wait > 0:
                    time.sleep(wait)
            changed.clear()     # this request covers everything transcribed until now
            last_request = time.monotonic()
            try:
                self.generate_response_from_transcript()
            except Exception as exception:  # pylint: disable=broad-except
                logger.exception(f'Response generation failed: {exception}')

    def update_response_interval(self, interval):
        """Change the interval for pinging LLM
        """
        self.llm_response_interval = interval


class OpenAIResponder(GPTResponder):
    """Uses OpenAI (or any OpenAI-compatible provider set as Base URL) for Chat Inference"""
    settings_section = 'OpenAI'

    def __init__(self,
                 config: dict,
                 convo: conversation.Conversation,
                 response_file_name: str,
                 save_to_file: bool = False,
                 base_url: str = None):
        del base_url
        super().__init__(config=config,
                         convo=convo,
                         save_to_file=save_to_file,
                         file_name=response_file_name)
        logger.info(f'Configured OpenAI response generation. Model: {self.model}')


class TogetherAIResponder(GPTResponder):
    """Uses TogetherAI for Chat Inference"""
    settings_section = 'Together'

    def __init__(self,
                 config: dict,
                 convo: conversation.Conversation,
                 response_file_name: str,
                 save_to_file: bool = False):
        super().__init__(config=config,
                         convo=convo,
                         save_to_file=save_to_file,
                         file_name=response_file_name)
        logger.info(f'Configured Together AI response generation. Model: {self.model}')


class InferenceResponderFactory:
    """Factory class to get the appropriate Inference Provider / GPT Provider
    """
    def get_responder_instance(self,
                               provider: InferenceEnum,
                               config: dict,
                               convo: conversation.Conversation,
                               response_file_name: str,
                               save_to_file: bool = False,
                               ) -> GPTResponder:
        """Get the appropriate Inference Provider class instance"""
        if not isinstance(provider, InferenceEnum):
            raise TypeError('InferenceResponderFactory: provider should be an instance of InferenceEnum')

        if provider == InferenceEnum.OPENAI:
            return OpenAIResponder(config=config,
                                   convo=convo,
                                   save_to_file=save_to_file,
                                   response_file_name=response_file_name)
        elif provider == InferenceEnum.TOGETHER:
            return TogetherAIResponder(config=config,
                                       convo=convo,
                                       save_to_file=save_to_file,
                                       response_file_name=response_file_name)

        raise ValueError("Unknown Inference Provider type")
