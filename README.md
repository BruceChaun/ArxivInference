# DS-GA 1005

### Procedure

#### Data Processing

1. We use [arxiv sanity preserver](https://github.com/karpathy/arxiv-sanity-preserver) to fetch and download papers. Make sure run the commands first. By default, clone arxiv sanity preserver into the same directory, otherwise, don't forget to change ```arxiv_sanity_pdf_path```.

2. Install [detex](https://github.com/pkubowicz/opendetex). We will use this open package to remove the LaTeX format.

3. After downloading the pdf files, I recommend you change ```user_agent``` to your own in _arxiv.py_ file because sometimes arxiv will still block the existent one. Run ```python arxiv.py``` script to process the files.

4. All the files are processed and stored in _data_ folder.
