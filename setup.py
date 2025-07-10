from setuptools import setup, find_packages

setup(
    name='counseleval',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'fire',
        'tqdm',
        'moviepy',
        'opencv-python',
        'librosa',
        'mediapipe',
        'openface',
        'opensmile',
        'psutil',
    ],
    entry_points={
        'console_scripts': [
            'counseleval=counseleval.cli:main',
        ],
    },
)

