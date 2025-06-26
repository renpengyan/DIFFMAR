import os
import subprocess

def run(script, args):
    cmd = ['python', script] + args
    print("Running:", ' '.join(cmd))
    subprocess.run(cmd, check=True)

def main():
    base = 'data/train'
    run('tools/generate_xt.py', [os.path.join(base, 'x0'), os.path.join(base, 'mask'), os.path.join(base, 'xT')])
    run('tools/generate_xli.py', [os.path.join(base, 'xT'), os.path.join(base, 'mask'), os.path.join(base, 'xLI')])

if __name__ == '__main__':
    main()
