# -*- coding: utf-8 -*-
def main():  # nocover
    import wbia_segmentation

    print('Looks like the imports worked')
    print('wbia_segmentation = {!r}'.format(wbia_segmentation))
    print('wbia_segmentation.__file__ = {!r}'.format(wbia_segmentation.__file__))
    print('wbia_segmentation.__version__ = {!r}'.format(wbia_segmentation.__version__))


if __name__ == '__main__':
    """
    CommandLine:
       python -m wbia_segmentation
    """
    main()