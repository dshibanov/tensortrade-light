
from tensortrade.data.feed.core import Stream


def test_boolean():

    s = Stream.source(["hello", "my", "name", "is", "slim", "shady"], dtype="string")

    s.startswith("m").invert()
