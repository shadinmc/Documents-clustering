# Document Clustering System
This project provides a backend + frontend application for topic-based document clustering and file organization.

The system:
- Scans a source folder (local path or any server-mounted path accessible by the machine).
- Extracts text from supported files.
- Clusters files by topic using TF-IDF + KMeans.
- Creates cluster folders automatically.
- Copies or moves files into the new folder structure.

## Supported file types
- Text-like files: `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.html`, `.htm`
- Code files: `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`
- Documents: `.pdf`, `.docx`

## Run locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start server:
   ```bash
   python code.py
   ```
3. Open UI:
   - `http://localhost:8000`

## API
### `POST /api/cluster`
Request example:
```json
{
  "source_path": "C:\\docs\\incoming",
  "destination_path": "C:\\docs\\clustered",
  "mode": "copy",
  "recursive": true,
  "num_clusters": 5
}
```

`mode` can be `copy` or `move`.
If `destination_path` is not provided, output is created as `<source_name>_clustered` next to the source folder.
