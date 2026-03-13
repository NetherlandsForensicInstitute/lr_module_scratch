import shutil
from pathlib import Path

import pandas as pd
from lir.aggregation import Aggregation, ContextAwareDict, config_parser, partial, pop_field
from lir.data.io import search_path
from lir.util import check_type


class CopyCSV(Aggregation):
    """Aggregation that copies a CSV file from a source location to a target location, optionally selecting columns.

    Attributes
    ----------
        source_file (str): The path to the source CSV file that should be copied.
        target_dir (str): The directory where the new CSV file will be saved. Given by the config_parser, meaning
                            it is not set by the user in the configuration.
        columns (list[str]): A list of column names to copy from the source CSV. If empty, all columns will be copied.
        new_file_name (str): The name of the new CSV file. If empty, the original file name will be used.
    """

    def __init__(self, source_file: str, target_dir: str, columns: list[str], new_file_name: str | None):
        self.source_file = search_path(Path(source_file))
        self.target_dir = Path(target_dir)
        self.columns = columns
        if new_file_name is None:
            self.new_file_name = self.target_dir / self.source_file.name
        else:
            self.new_file_name = self.target_dir / new_file_name

    def report(self, data) -> None:
        """Do nothing. Required by parent class."""
        pass

    def close(self):
        """Close the aggregation and perform any necessary cleanup.

        This method has the logic of this class. It copies the CSV file from the source to the target location,
        optionally selecting specific columns if they are specified.
        """
        if self.columns:
            df = pd.read_csv(self.source_file)
            df[self.columns].to_csv(self.new_file_name, index=False)
        else:
            shutil.copy(self.source_file, self.new_file_name)


@config_parser()
def copy_csv(config: ContextAwareDict, output_dir: str) -> CopyCSV:
    """Parse the configuration for the CopyCSV aggregation and return an instance of it."""
    source_file = pop_field(config, "file")
    columns = pop_field(config, "columns", default=[], validate=partial(check_type, list))
    new_file_name = pop_field(config, "new_file_name", required=False)
    return CopyCSV(source_file, output_dir, columns=columns, new_file_name=new_file_name)
