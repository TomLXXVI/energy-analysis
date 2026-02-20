from pathlib import Path

from energy_analysis import batch_plot_directory

src_path = Path("../data")
dest_path = Path("../pdf_output")

created = batch_plot_directory(src_path, out_dir=dest_path)
for path in created:
    print(path)
