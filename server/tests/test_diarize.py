from meetings_server.diarize import Turn, assign_speakers
from meetings_server.schema import Segment, Transcript, Word


def _w(start, end, text):
    return Word(start=start, end=end, text=text)


def test_segments_split_at_speaker_change_and_renamed():
    t = Transcript(
        segments=[
            Segment(
                start=0.0,
                end=4.0,
                text="Hello there. Hi back.",
                words=[_w(0.0, 0.5, " Hello"), _w(0.5, 1.0, " there."), _w(2.1, 2.5, " Hi"), _w(2.5, 3.0, " back.")],
            ),
            Segment(start=5.0, end=6.0, text="No words here."),
        ]
    )
    turns = [Turn(0.0, 2.0, "SPEAKER_07"), Turn(2.0, 4.0, "SPEAKER_03"), Turn(4.8, 6.5, "SPEAKER_07")]
    out = assign_speakers(t, turns)

    assert out.diarized
    assert [(s.speaker, s.text) for s in out.segments] == [
        ("Speaker 1", "Hello there."),
        ("Speaker 2", "Hi back."),
        ("Speaker 1", "No words here."),
    ]
    assert out.segments[1].start == 2.1
    assert out.speakers == ["Speaker 1", "Speaker 2"]


def test_word_outside_any_turn_gets_nearest_speaker():
    t = Transcript(segments=[Segment(start=10.0, end=10.4, text="Ok", words=[_w(10.0, 10.4, " Ok")])])
    out = assign_speakers(t, [Turn(0.0, 3.0, "A"), Turn(9.0, 9.8, "B")])
    assert out.segments[0].speaker == "Speaker 1"
    assert out.segments[0].words[0].speaker == "Speaker 1"
