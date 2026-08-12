import unittest
from unittest import mock

from melee import console
from melee import slippstream


class FakeContext:
    def __init__(self):
        self.calls = []

    def get_start_method(self):
        return "fake"

    def Pipe(self, duplex):
        self.calls.append(("Pipe", duplex))
        return mock.sentinel.parent_buffer, mock.sentinel.worker_buffer

    def Event(self):
        self.calls.append(("Event",))
        return mock.sentinel.shutdown

    def Process(self, *, target, kwargs):
        self.calls.append(("Process", target, kwargs))
        return mock.sentinel.worker


class SlippstreamMultiprocessingContext(unittest.TestCase):
    @mock.patch.object(slippstream.platform, "system", return_value="Linux")
    @mock.patch.object(slippstream.mp, "get_context")
    def test_linux_default_is_fork(self, get_context, _system):
        slippstream._default_multiprocessing_context()
        get_context.assert_called_once_with("fork")

    @mock.patch.object(slippstream.platform, "system", return_value="Darwin")
    @mock.patch.object(slippstream.mp, "get_context")
    def test_non_linux_uses_platform_default(self, get_context, _system):
        slippstream._default_multiprocessing_context()
        get_context.assert_called_once_with()

    def test_client_uses_one_injected_context_for_all_primitives(self):
        context = FakeContext()
        client = slippstream.SlippstreamClient(
            "127.0.0.1",
            51441,
            multiprocessing_context=context,
        )

        self.assertIs(client._buffer, mock.sentinel.parent_buffer)
        self.assertIs(client._shutdown, mock.sentinel.shutdown)
        self.assertIs(client._worker, mock.sentinel.worker)
        self.assertEqual(context.calls[0], ("Pipe", False))
        self.assertEqual(context.calls[1], ("Event",))
        self.assertEqual(context.calls[2][0], "Process")
        self.assertIs(context.calls[2][2]["buffer"], mock.sentinel.worker_buffer)
        self.assertIs(context.calls[2][2]["shutdown"], mock.sentinel.shutdown)

    @mock.patch.object(console, "SlippstreamClient")
    def test_console_forwards_context(self, client):
        context = mock.sentinel.context
        console.Console(
            is_remote=True,
            multiprocessing_context=context,
        )
        client.assert_called_once_with(
            "127.0.0.1",
            51441,
            multiprocessing_context=context,
        )


if __name__ == "__main__":
    unittest.main()
