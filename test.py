import app

def test_main_runs_without_error():
  try:
    app.main()
  except Exception as e:
    assert False, f"main() raised an exception: {e}"

def test_has_main_function():
  assert hasattr(app,'main'), "app.py does not have a main() function"
