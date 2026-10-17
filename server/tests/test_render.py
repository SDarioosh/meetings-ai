from meetings_server.render import fmt_duration, transcript_to_text
from meetings_server.schema import Segment, Transcript


def test_same_speaker_segments_merge_into_one_turn(transcript):
    lines = transcript_to_text(transcript).splitlines()
    assert lines[0] == (
        "[00:00:00] Speaker 1: Okay, let's start. This is the launch planning sync for the Falcon dashboard. "
        "Priya, can you give us the status on the export feature?"
    )
    assert lines[1].startswith("[00:00:14] Speaker 2: Sure.")


def test_unlabelled_segments_stay_on_separate_lines():
    t = Transcript(segments=[Segment(start=0, end=2, text="One."), Segment(start=65, end=67, text="Two.")])
    assert transcript_to_text(t) == "[00:00:00] One.\n[00:01:05] Two.\n"


def test_fmt_duration():
    assert fmt_duration(73) == "1m 13s"
    assert fmt_duration(3725) == "1h 02m"
    assert fmt_duration(0) == ""
