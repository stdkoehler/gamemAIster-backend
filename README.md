```shell
C:\Users\Stefan\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.\.venv\Scrips\activate
pip install pip-tools
```

```shell
pip-compile --extra dev --no-emit-index-url --resolver=backtracking pyproject.toml --verbose
pip-sync requirements.txt
```