import sys
packages = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn', 'torch', 'torchvision', 'torchinfo', 'thop', 'psutil', 'joblib']
for p in packages:
    try:
        mod = __import__(p)
        print(f"{p}=={getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"{p}: uninstalled")
