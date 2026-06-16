def skip_package_dir(package):
    return package == "triton" or package.startswith("triton.")


def get_package_dir():
    return {}
