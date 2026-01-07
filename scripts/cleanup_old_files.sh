#!/bin/bash
# 清理项目中的旧版本和临时测试文件

echo "========================================"
echo "清理旧版本文件"
echo "========================================"
echo ""

BACKUP_DIR="backup_old_files_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "备份目录: $BACKUP_DIR"
echo ""

# 要清理的文件列表
OLD_FILES=(
    "app/ui/main.py"           # 旧版 UI (使用 main_v1.py)
    "run.py"                    # 旧版启动脚本 (使用 run_v1.py 或 start.sh)
    "test_hunyuan_ocr.py"      # 临时测试脚本
    "test_index_build.py"      # 临时测试脚本
    "test_ocr_api.py"          # 临时测试脚本
    "test_ocr_init.py"         # 临时测试脚本
    "test_ocr_with_image.py"   # 临时测试脚本
    "test_pdf_import_fix.py"   # 临时测试脚本
    "test_system_v1.py"        # 临时测试脚本
    "test_v1_smoke.py"         # 临时测试脚本
    "diagnose_system.py"       # 临时诊断脚本
    "verify_index_fixes.py"    # 临时验证脚本
)

# 移动文件到备份目录
for file in "${OLD_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  移动: $file"
        mv "$file" "$BACKUP_DIR/"
    fi
done

echo ""
echo "========================================"
echo "清理完成"
echo "========================================"
echo ""
echo "备份位置: $BACKUP_DIR"
echo ""
echo "保留的主要文件:"
echo "  • app/ui/main_v1.py  - 当前 UI"
echo "  • run_v1.py          - 启动脚本"
echo "  • start.sh           - 快速启动"
echo "  • scripts/*.sh       - 系统脚本"
echo ""
echo "如果需要恢复文件:"
echo "  mv $BACKUP_DIR/[filename] ."
echo ""
