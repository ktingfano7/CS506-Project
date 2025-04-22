import app

def test_run_main():
  try:
    app.main()
  except Exception as e:
    assert False, f"main() raised an exception: {e}"

def test_check_main():
  assert hasattr(app,'main'), "app.py does not have a main() function"
