"""
测试报告生成工具占位
"""
def generate_report(results, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(results))
    return out_path
