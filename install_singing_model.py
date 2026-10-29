"""Download official model releases into workspace cache. No audio upload."""
from pathlib import Path
import argparse
import hashlib
import shutil
import zipfile
import requests

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'work'/'model-cache'

def verify(path,algorithm,expected):
    with path.open('rb') as f:
        actual=hashlib.file_digest(f,algorithm).hexdigest()
    if actual!=expected:
        raise RuntimeError(f'模型校验失败：{path}')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all',action='store_true',help='Also install experimental ROSVOT (~584 MB).')
    args=parser.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    target=CACHE/'panns-cnn6.pth'
    digest='e25e26b84585b14c7754c91e48efc9be'
    if not target.exists():
        temp=target.with_suffix('.download')
        with requests.get('https://zenodo.org/records/3987831/files/Cnn6_mAP%3D0.343.pth?download=1',stream=True,timeout=120) as response:
            response.raise_for_status()
            with temp.open('wb') as f:
                for chunk in response.iter_content(1024*1024):
                    f.write(chunk)
        verify(temp,'md5',digest)
        temp.replace(target)
    verify(target,'md5',digest)
    if args.all:
        import gdown
        archive=ROOT/'work'/'research'/'rosvot-checkpoints.zip'
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():
            temp=archive.with_suffix('.download')
            gdown.download(id='1JNtNT37KiLq9uFQqHk7JFs-3trxd3bRh',output=str(temp),use_cookies=False)
            verify(temp,'sha256','b6055e81315b93415c9bd7fc48e10a28a3da1bea960cab7385483bd7443ba852')
            temp.replace(archive)
        verify(archive,'sha256','b6055e81315b93415c9bd7fc48e10a28a3da1bea960cab7385483bd7443ba852')
        # Explicit allowlist prevents archive paths from escaping the cache.
        with zipfile.ZipFile(archive) as z:
            for model in ('rmvpe','rosvot','rwbd'):
                for filename in (('model.pt',) if model=='rmvpe' else ('model.pt','config.yaml')):
                    name=f'checkpoints/{model}/{filename}'
                    destination=CACHE/'rosvot'/name
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(name) as src,destination.open('wb') as dst:
                        shutil.copyfileobj(src,dst)
    print('Model cache verified. Ready.')

if __name__=='__main__':
    main()
