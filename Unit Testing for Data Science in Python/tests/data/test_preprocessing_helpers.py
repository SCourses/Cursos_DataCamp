import sys
import os

# Agregar el directorio raíz de tu proyecto al sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Import the function convert_to_int()
from src.data.preprocessing_helpers import convert_to_int

# Complete the unit test name by adding a prefix
def test_on_string_with_one_comma():
  # Complete the assert statementS
  assert convert_to_int("2,082")==2082


def test_on_string_with_one_comma_1():
    test_argument = "2,082"
    expected = 2082
    actual = convert_to_int(test_argument)
    # Format the string with the actual return value
    message = "convert_to_int('2,081') should return the int 2081, but it actually returned {0}".format(actual)
    # Write the assert statement which prints message on failure
    assert actual==expected,message

