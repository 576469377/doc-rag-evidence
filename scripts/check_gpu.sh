#!/bin/bash
# 显示 GPU 内存使用情况

echo "========================================"
echo "GPU 内存使用状态"
echo "========================================"
echo ""

nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv | \
    awk 'NR==1 {print; next} {
        split($0, a, ",");
        gpu_id = a[1];
        used = a[3];
        free = a[4];
        total = a[5];
        gsub(/ MiB/, "", used);
        gsub(/ MiB/, "", free);
        gsub(/ MiB/, "", total);
        pct = int(used * 100 / total);
        
        status = "🟢 Free";
        if (pct > 90) status = "🔴 Full";
        else if (pct > 70) status = "🟡 Busy";
        
        printf "GPU %s: %s %5d / %5d MiB (%2d%%) %s\n", gpu_id, status, used, total, pct, a[2];
    }'

echo ""
echo "========================================"
echo "建议使用:"
echo "  • GPU 2 或 GPU 3 (空闲)"
echo "========================================"
