import os
import re

def fix_lietorch_source(root_dir):
    # 1. 定义需要修改的文件后缀
    extensions = ('.cpp', '.cu', '.h', '.hpp')
    
    # 2. 定义替换规则
    replacements = [
        # 规则 A: 将 Tensor.type() 替换为 Tensor.scalar_type()
        # 匹配类似 X.type(), tensor.type(), self->type() 等
        (re.compile(r'(\w+)\.type\(\)'), r'\1.scalar_type()'),
        
        # 规则 B: 修正 dispatch.h 中的宏定义错误
        # 将 at::ScalarType _st = ::detail::scalar_type(the_type); 
        # 改为 at::ScalarType _st = the_type; (因为此时 the_type 已经是 scalar_type 了)
        (re.compile(r'at::ScalarType _st = ::detail::scalar_type\(the_type\);'), 
         r'at::ScalarType _st = the_type;')
    ]

    print(f"开始扫描目录: {root_dir}")

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(subdir, file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                new_content = content
                for pattern, subst in replacements:
                    new_content = pattern.sub(subst, new_content)

                if new_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"已修复: {file_path}")

if __name__ == "__main__":
    # 指向你的 lietorch 源码路径
    target_dir = "./thirdparty/lietorch"
    
    if os.path.exists(target_dir):
        fix_lietorch_source(target_dir)
        print("\n所有过时 API 已替换完成。")
        print("提示：请在重新编译前运行 'rm -rf build/' 清理缓存。")
    else:
        print(f"错误：找不到目录 {target_dir}，请确认脚本运行位置。")