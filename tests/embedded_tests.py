"""Native output and lifecycle behavior used by embedded callers."""

import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import jq
import pytest


def render(program, text, **options):
    with closing(jq.compile(program).input_text(text).text_items(**options)) as items:
        return "".join(item + "\n" for item in items)


@pytest.mark.parametrize("pretty", [False, True])
def test_native_output_preserves_large_integer_and_decimal(pretty):
    source = '{"integer":9007199254740993,"decimal":0.12345678901234567890123456789}'

    output = render(".", source, pretty=pretty)

    assert "9007199254740993" in output
    assert "0.12345678901234567890123456789" in output
    assert output == (
        '{\n  "integer": 9007199254740993,\n'
        '  "decimal": 0.12345678901234567890123456789\n}\n'
        if pretty else source + "\n"
    )


@pytest.mark.parametrize("pretty", [False, True])
def test_raw_output_preserves_nul_unicode_and_non_string_values(pretty):
    output = render(".[]", '["a\\u0000é",true,null,{"x":1}]', pretty=pretty, raw=True)

    assert output == "a\0é\ntrue\nnull\n" + ('{\n  "x": 1\n}\n' if pretty else '{"x":1}\n')


def test_edit_preserves_four_mib_string():
    payload = "a" * (4 * 1024 * 1024)
    source = json.dumps({"payload": payload, "revision": 1})

    output = render(".revision += 1", source)

    assert json.loads(output) == {"payload": payload, "revision": 2}


@pytest.mark.parametrize(("program", "text", "expected"), [
    ("input", "1\n2\n3\n4", ["2", "4"]),
    ("inputs", "1\n2\n3", ["2", "3"]),
    ("reduce inputs as $x (.; . + $x)", "1\n2\n3", ["6"]),
    ("[., input, inputs]", "1\n2\n3\n4", ["[1,2,3,4]"]),
    ('try inputs catch "ignored"', "1\n2\n{", ["2", '"ignored"']),
    (".", "", []),
    ("empty", "{}", []),
], ids=["input", "inputs", "reduce", "mixed", "catch", "empty-input", "empty-output"])
def test_input_callback_consumes_the_same_stream(program, text, expected):
    assert list(jq.compile(program).input_text(text).text_items()) == expected


@pytest.mark.parametrize(("program", "text", "expected"), [
    (".", "", ["[]"]),
    ("length", "", ["0"]),
    (".", "1\n2\n3", ["[1,2,3]"]),
    ("[., inputs]", "1\n2", ["[[1,2]]"]),
])
def test_slurp_evaluates_once_with_input_callback_at_eof(program, text, expected):
    assert list(jq.compile(program).input_text(text, slurp=True).text_items()) == expected


@pytest.mark.parametrize("program", ["halt", "halt_error(0)"])
def test_successful_halt_stops_later_inputs(program):
    assert render(program, "null\n{") == ""


def test_halt_error_raises_after_partial_output():
    result = jq.compile('., "stop" | if . == "stop" then halt_error(7) else . end')
    iterator = result.input_text("1\n2").text_items()

    assert next(iterator) == "1"
    with pytest.raises(ValueError, match="exit code 7"):
        next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)


@pytest.mark.parametrize("program", [".", "inputs"])
def test_uncaught_late_parse_error_fails(program):
    with pytest.raises(ValueError):
        render(program, "1\n2\n{")


def test_close_is_idempotent_and_returns_state_with_a_fresh_input_callback():
    program = jq.compile("., inputs")
    first = program.input_text("1\n2\n3").text_items()
    assert next(first) == "1"

    first.close()
    first.close()

    with pytest.raises(StopIteration):
        next(first)
    assert list(program.input_text("4\n5").text_items()) == ["4", "5"]


def test_abandoned_iterator_does_not_retain_a_dangling_input_callback():
    program = jq.compile("., inputs")
    first = program.input_text("1\n2").text_items()
    assert next(first) == "1"
    del first
    gc.collect()

    assert list(program.input_text("3\n4").text_items()) == ["3", "4"]


def test_two_live_iterators_have_independent_input_callbacks():
    program = jq.compile("., inputs")
    with closing(program.input_text("1\n2").text_items()) as first:
        with closing(program.input_text("3\n4").text_items()) as second:
            assert next(first) == "1"
            assert next(second) == "3"
            assert list(first) == ["2"]
            assert list(second) == ["4"]


def test_compile_and_runtime_failures_do_not_poison_reused_states():
    for _ in range(10):
        with pytest.raises(ValueError):
            jq.compile("invalid(")
    program = jq.compile('if . < 0 then error("failed") else . + $offset end', args={"offset": 2})
    with pytest.raises(ValueError, match="failed"):
        list(program.input_value(-1).text_items())

    assert list(program.input_value(3).text_items()) == ["5"]


def test_shared_program_is_safe_across_threads():
    program = jq.compile("., inputs")

    def evaluate(index):
        return list(program.input_text(f"{index}\n{index + 1}").text_items())

    with ThreadPoolExecutor(max_workers=8) as workers:
        actual = list(workers.map(evaluate, range(200)))

    assert actual == [[str(index), str(index + 1)] for index in range(200)]


def test_concurrent_compilers_keep_error_callbacks_and_args_separate():
    def evaluate(index):
        if index % 2:
            with pytest.raises(ValueError):
                jq.compile("invalid(")
            return None
        program = jq.compile(". + $offset", args={"offset": index})
        return list(program.input_value(1).text_items())

    with ThreadPoolExecutor(max_workers=8) as workers:
        actual = list(workers.map(evaluate, range(200)))

    assert actual == [None if index % 2 else [str(index + 1)] for index in range(200)]


def test_same_iterator_serializes_concurrent_next_calls():
    iterator = jq.compile("range(0; 200)").input_value(None).text_items()
    with ThreadPoolExecutor(max_workers=8) as workers:
        actual = list(workers.map(lambda _: next(iterator), range(200)))

    iterator.close()
    assert sorted(map(int, actual)) == list(range(200))


def test_evaluation_releases_gil_for_other_python_work():
    program = jq.compile("reduce range(0; 2000000) as $i (0; . + $i)")
    ticks = 0

    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(lambda: list(program.input_value(None).text_items()))
        while not pending.done():
            ticks += 1
            time.sleep(0.005)
        result = pending.result()

    assert result == ["1999999000000"]
    assert ticks >= 3
