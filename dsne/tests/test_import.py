def test_import():
    import dsne

    assert dsne.__version__ is not None
    assert dsne.__version__ != "0.0.0"
    assert len(dsne.__version__) > 0


def test_import_st_dsne():
    from dsne import dsne  # noqa
    from dsne import dsne_approximate  # noqa
