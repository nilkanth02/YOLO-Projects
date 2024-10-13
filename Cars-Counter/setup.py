from setuptools import setup, find_packages

setup(
    name='object_detection_yolo',  # Your package name
    version='0.1.0',  # Initial version
    author='Your Name',  # Your name
    author_email='your_email@example.com',  # Your email
    description='YOLO object detection implementation',  # Package description
    long_description=open('README.md').read(),  # Ensure you have a README.md file
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/object_detection_yolo',  # Your GitHub repo URL
    packages=find_packages(),  # Automatically find packages
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Supported Python versions
    install_requires=[
        'numpy',  # Add other dependencies your project needs
        "opencv-python==4.10.0.82",
        'tensorflow',  # or 'torch', depending on your implementation
        'lap',  # Include LAP if it’s a dependency
    ],
)
