# test_data_and_fixtures.py
import pytest


class TestData:
    __test__ = False

    def __init__(self):
        self.grade = 0


@pytest.fixture(scope="session")
def global_data(request):
    data = TestData()

    def finalize():
        result = final_grade(data)
        print(result, end="", flush=True)

    request.addfinalizer(finalize)
    return data


def final_grade(test_data_instance):
    return f"Overall grade: {test_data_instance.grade}\n"
