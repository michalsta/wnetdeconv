
def test_import():
    import wnetdeconv
    wnetdeconv.hello()
    assert wnetdeconv.wnetdeconv_cpp.hello() == 0