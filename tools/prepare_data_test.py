import os
import subprocess

def run_script(script, args):
    cmd = ['python', script] + args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    base_dir = 'data/test'
    x0_dir = os.path.join(base_dir, 'x0')
    mask_dir = os.path.join(base_dir, 'mask')
    xT_dir = os.path.join(base_dir, 'xT')
    xLI_dir = os.path.join(base_dir, 'xLI')

    # 自动创建输出文件夹
    os.makedirs(xT_dir, exist_ok=True)
    os.makedirs(xLI_dir, exist_ok=True)

    # 调用生成 xT 的脚本
    run_script('tools/generate_xt.py', [x0_dir, mask_dir, xT_dir])

    # 调用生成 xLI 的脚本
    run_script('tools/generate_xli.py', [xT_dir, mask_dir, xLI_dir])

if __name__ == '__main__':
    main()
