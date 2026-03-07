from enum import Enum

class Password_Exceeded(Enum):
    EXCEED_MIN_LENGTH = "Password must exceed minimum length of 8 characters."
    EXCEED_MAX_LENGTH = "Password must not exceed maximum length of 128 characters."
    EXCEED_REQUIRE_UPPERCASE = "Password must contain at least one uppercase letter."
    EXCEED_REQUIRE_LOWERCASE = "Password must contain at least one lowercase letter."
    EXCEED_REQUIRE_DIGIT = "Password must contain at least one digit."
    EXCEED_REQUIRE_SPECIAL_CHAR = "Password must contain at least one special character."
    VALID = "Password is valid."
