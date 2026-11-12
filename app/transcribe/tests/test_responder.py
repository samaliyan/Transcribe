"""Replies from the AI provider: prompts, streaming, and errors worded for the user."""

import datetime
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai

from app.transcribe import gpt_responder, prompts
from app.transcribe.core import conversation as core_conversation

T0 = datetime.datetime(2026, 1, 1, 12, 0, 0)


def chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def make_config(api_key="sk-test", seed=None):
    return {
        "General": {"system_prompt": "Help me.", "summary_prompt": "Summarize.", "initial_convo": seed or {},
                    "llm_response_interval": 5, "chat_inference_provider": "openai",
                    "default_prompt_preamble": "Reply to:", "default_prompt_epilogue": ""},
        "OpenAI": {"api_key": api_key, "base_url": "https://example.test/v1", "ai_model": "model-x",
                   "response_lang": "English", "response_request_timeout_seconds": 20,
                   "summarize_request_timeout_seconds": 30, "temperature": None, "reasoning_effort": None},
    }


class ResponderCase(unittest.TestCase):
    def setUp(self):
        self.table = MagicMock()
        self.table.insert_conversation.side_effect = range(100, 200)
        fake_db = MagicMock()
        fake_db.get_invocation_id.return_value = 1
        fake_db.get_object.return_value = self.table
        for target in (core_conversation, gpt_responder):
            patcher = patch.object(target, "appdb", return_value=fake_db)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make(self, api_key="sk-test", seed=None):
        config = make_config(api_key, seed)
        with patch.object(core_conversation.configuration, "Config", return_value=SimpleNamespace(data=config)):
            convo = core_conversation.Conversation(context=SimpleNamespace())
        self.client = MagicMock()
        module = SimpleNamespace(OpenAI=MagicMock(return_value=self.client))
        responder = gpt_responder.GPTResponder(config=config, convo=convo, file_name="unused.txt",
                                               openai_module=module)
        return responder, convo, module


class TestPrompts(ResponderCase):
    def test_the_model_is_told_who_said_what(self):
        responder, convo, _ = self.make()
        convo.update_conversation("Speaker", "What is your notice period?", T0)
        convo.update_conversation("You", "Let me think.", T0 + datetime.timedelta(seconds=2))
        convo.update_conversation("Speaker", "Take your time.", T0 + datetime.timedelta(seconds=4))
        messages = prompts.create_multiturn_prompt(convo.get_merged_conversation_response(length=12))
        self.assertEqual(messages[0], {"role": "system", "content": "Help me. Respond exclusively in English."})
        self.assertEqual(messages[1:], [{
            "role": "user",
            "content": "Other: What is your notice period?\nMe: Let me think.\nOther: Take your time."}])

    def test_no_seed_conversation_is_needed_and_one_from_the_settings_is_kept(self):
        _, convo, _ = self.make()
        self.assertFalse(convo.has_spoken_rows())
        seed = {"first": {"role": "You", "content": "I am V."}, "second": {"role": "assistant", "content": "Hello V."}}
        _, seeded, _ = self.make(seed=seed)
        self.assertFalse(seeded.has_spoken_rows())
        seeded.update_conversation("Speaker", "hi", T0 + datetime.timedelta(days=400))
        seeded.update_conversation("Speaker", "hi there", T0, update_previous=True)
        messages = prompts.create_multiturn_prompt(seeded.get_merged_conversation_response(length=12))
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "Me: I am V.")
        self.assertEqual(messages[3]["content"], "Other: hi there")

    def test_the_first_live_row_of_the_microphone_is_updated_in_place(self):
        _, convo, _ = self.make()
        convo.update_conversation("You", "hel", T0)
        convo.update_conversation("You", "hello", T0, update_previous=True)
        self.assertEqual([row[0] for row in convo.transcript_data["You"]], ["You: [hello]\n\n"])

    def test_brackets_of_old_prompts_are_removed_from_replies(self):
        self.assertEqual(prompts.strip_brackets("[Say yes.]"), "Say yes.")
        self.assertEqual(prompts.strip_brackets("[Say y"), "Say y")
        self.assertEqual(prompts.strip_brackets("Use a[0] and b[1]"), "Use a[0] and b[1]")
        self.assertEqual(prompts.split_row("Speaker: [a: b]\n\n"), ("Speaker", "a: b"))

    def test_the_reply_language_and_prompt_can_change_while_running(self):
        responder, convo, _ = self.make()
        responder.config["OpenAI"]["response_lang"] = "persian"
        responder.config["General"]["system_prompt"] = "Be brief."
        convo.refresh_system_prompt()
        convo.update_conversation("Speaker", "hi", T0)
        messages = prompts.create_multiturn_prompt(convo.get_merged_conversation_response())
        self.assertEqual(messages[0]["content"], "Be brief. Respond exclusively in Persian.")


class TestRequests(ResponderCase):
    def test_a_streamed_reply_is_collected_and_stored_for_its_row(self):
        responder, convo, _ = self.make()
        convo.update_conversation("Speaker", "question?", T0)
        self.client.chat.completions.create.return_value = iter(
            [chunk("Say "), chunk(None), SimpleNamespace(choices=[]), chunk("yes.")])
        with patch.object(responder, "_insert_response_in_db") as insert:
            self.assertEqual(responder.generate_response_from_transcript_no_check(), "Say yes.")
        insert.assert_called_once_with(100, "Say yes.")
        self.assertEqual(responder.response, "Say yes.")
        self.assertEqual(responder.status, gpt_responder.STATUS_READY)
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual((request["model"], request["stream"], request["timeout"]), ("model-x", True, 20))

    def test_asking_twice_without_new_speech_works(self):
        """The newest row used to be the previous reply, which has no database id."""
        responder, convo, _ = self.make()
        convo.update_conversation("Speaker", "question?", T0)
        for _ in range(2):
            self.client.chat.completions.create.return_value = iter([chunk("ok")])
            with patch.object(responder, "_insert_response_in_db"):
                self.assertEqual(responder.generate_response_from_transcript_no_check(), "ok")

    def test_without_a_key_nothing_is_sent_and_the_status_says_so(self):
        responder, convo, module = self.make(api_key="API_KEY")
        module.OpenAI.assert_not_called()
        self.assertEqual(responder.status, gpt_responder.STATUS_NO_KEY)
        convo.update_conversation("Speaker", "question?", T0)
        self.assertIsNone(responder.generate_response_from_transcript_no_check())
        self.assertEqual(responder.status, gpt_responder.STATUS_NO_KEY)
        with self.assertRaises(gpt_responder.ResponderError):
            responder.summarize()

    def test_a_new_key_from_the_settings_is_used_without_a_restart(self):
        responder, _, module = self.make(api_key="API_KEY")
        responder.config["OpenAI"]["api_key"] = " sk-new "
        responder.config["OpenAI"]["ai_model"] = "model-y"
        responder.configure()
        self.assertEqual(module.OpenAI.call_args.kwargs["api_key"], "sk-new")
        self.assertEqual((responder.status, responder.model), (gpt_responder.STATUS_READY, "model-y"))

    def test_a_failure_is_explained_and_the_next_request_can_succeed(self):
        responder, convo, _ = self.make()
        convo.update_conversation("Speaker", "question?", T0)
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        self.client.chat.completions.create.side_effect = openai.APIConnectionError(request=request)
        self.assertIsNone(responder.generate_response_from_transcript_no_check())
        self.assertEqual(responder.status, gpt_responder.STATUS_ERROR)
        self.assertIn("could not be reached", responder.last_error)
        self.client.chat.completions.create.side_effect = None
        self.client.chat.completions.create.return_value = iter([chunk("ok")])
        with patch.object(responder, "_insert_response_in_db"):
            self.assertEqual(responder.generate_response_from_transcript_no_check(), "ok")
        self.assertIsNone(responder.last_error)

    def test_nothing_is_sent_before_anything_was_said(self):
        responder, _, _ = self.make()
        self.assertIsNone(responder.generate_response_from_transcript_no_check())
        self.client.chat.completions.create.assert_not_called()
        self.assertEqual(responder.last_error, gpt_responder.NOTHING_TO_ANSWER_MESSAGE)

    def test_summary_uses_what_was_said(self):
        responder, convo, _ = self.make()
        convo.update_conversation("Speaker", "We ship on Friday.", T0)
        self.client.chat.completions.create.return_value = iter([chunk("Ship Friday.")])
        self.assertEqual(responder.summarize(), "Ship Friday.")
        messages = self.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], "Summarize. Respond exclusively in English.")
        self.assertEqual(messages[1]["content"], "Other: We ship on Friday.")

    def test_selected_text_gets_its_own_request(self):
        responder, _, _ = self.make()
        self.client.chat.completions.create.return_value = iter([chunk("answer")])
        self.assertEqual(responder.generate_response_for_selected_text("what is TTL"), "answer")
        messages = self.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(messages[1], {"role": "user", "content": "Reply to:\n\nwhat is TTL"})


class TestErrorTexts(unittest.TestCase):
    def error(self, cls, status, body=None):
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        response = httpx.Response(status, request=request)
        return cls("failed", response=response, body=body)

    def test_each_kind_of_failure_tells_the_user_what_to_do(self):
        describe = gpt_responder.describe_error
        self.assertIn("rejected the API key", describe(self.error(openai.AuthenticationError, 401)))
        self.assertIn("region", describe(self.error(openai.PermissionDeniedError, 403)))
        self.assertIn('"m1"', describe(self.error(openai.NotFoundError, 404), "m1"))
        self.assertIn("limiting", describe(self.error(openai.RateLimitError, 429)))
        quota = openai.RateLimitError("You exceeded your current quota", response=httpx.Response(
            429, request=httpx.Request("POST", "https://example.test")), body=None)
        self.assertIn("no credit", describe(quota))
        self.assertIn("unsupported value", describe(self.error(
            openai.BadRequestError, 400, {"message": "unsupported value: temperature"})))
        self.assertEqual(describe(gpt_responder.ResponderError("as it is")), "as it is")

    def test_placeholders_are_not_keys(self):
        for value in (None, "", "  ", "API_KEY"):
            self.assertFalse(gpt_responder.has_api_key(value))
        self.assertTrue(gpt_responder.has_api_key("sk-abc"))
        self.assertEqual(gpt_responder.test_connection("API_KEY", "", "m")[0], False)


if __name__ == "__main__":
    unittest.main()
