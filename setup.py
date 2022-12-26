from setuptools import setup


setup(name='BERT_Family',
version='0.0.1',
description='BERT models summary',
author='Luka Jiang',
author_email='lukajiang1998@gmail.com',
url='https://github.com/kiangkiangkiang',
packages=['src/BERT_Family'],
# 表明當前模塊依賴哪些包，若環境中沒有，則會從pypi中下載安裝  
install_requires=['docutils>=0.3'],  
classifiers = [  
        # 發展時期,常見的如下  
        # 詳見https://pypi.org/pypi?%3Aaction=list_classifiers
        #   3 - Alpha  
        #   4 - Beta  
        #   5 - Production/Stable  
        'Development Status :: 1 - Planning',
  
        # 開發的目標用戶  
        'Intended Audience :: Science/Research',  
  
        # 屬於什麼類型  
        'Topic :: Education :: Testing',  
  
        # 許可證信息  
        'License :: OSI Approved :: MIT License',  
  
        # 目標 Python 版本  
        'Programming Language :: Python :: 3.9',  
    ],
license='MIT',
zip_safe=False)