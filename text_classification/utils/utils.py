from pathlib import Path

all_data_path = Path.cwd().parent/"data"


def get_parameter_value(value_dict, key, default_value=None):
    value = value_dict.get(key)
    if value is None:
        if default_value is None:
            return None
        else:
            return default_value


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
