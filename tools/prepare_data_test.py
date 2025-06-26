import os
import subprocess

def run_script(script_path, args):
    cmd = ['python', script_path] + args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    base = 'data/train'

    paths = {
        'x0': os.path.join(base, 'x0'),
        'xT': os.path.join(base, 'xT'),
        'mask': os.path.join(base, 'mask'),
        'xLI': os.path.join(base, 'xLI'),
    }

    # 确保输出文件夹存在
    os.makedirs(paths['xT'], exist_ok=True)
    os.makedirs(paths['xLI'], exist_ok=True)

    # 1. 生成 xT
    run_script('tools/generate_test_xt.py', [
        paths['x0'],
        paths['mask'],
        paths['xT']
    ])

    # 2. 生成 xLI
    run_script('tools/generate_test_xli.py', [
        paths['xT'],
        paths['mask'],
        paths['xLI']
    ])

if __name__ == '__main__':
    main()
