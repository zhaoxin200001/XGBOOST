# %%
# 导入Shiny框架的核心模块，用于构建Web应用
from shiny import App, ui, render, reactive
# 导入pandas库，用于数据处理和分析
import pandas as pd
# 导入numpy库，用于数值计算
import numpy as np
# 导入joblib库，用于加载和保存机器学习模型
import joblib
# 导入SHAP库，用于模型解释性分析
import shap
# 导入matplotlib绘图库
import matplotlib.pyplot as plt
# 导入io模块，用于处理字节流
import io
# 导入base64模块，用于编码图像数据
import base64
# 导入pathlib模块，用于处理文件路径
from pathlib import Path
# 导入正则表达式模块，用于字符串处理
import re
# 导入warnings模块，用于控制警告信息
import warnings



# 设置matplotlib的中文字体，确保图表能正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# 设置matplotlib正确显示负号
plt.rcParams['axes.unicode_minus'] = False

# 定义模型配置字典，简化为单个糖尿病预测模型
MODEL_CONFIG = {
    # 糖尿病随机森林模型的配置
    "diabetes_rf": {
        # 指定模型文件路径
        "file": "best_logistic_model.joblib",
        # 定义模型使用的特征列表
        "features": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    }
}

# 定义工具函数：清理文本以用作HTML ID
def sanitize_id(text):
    """清理文本以用作HTML ID"""
    # 使用正则表达式替换非字母数字字符为下划线
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(text))

# 定义安全加载模型的函数
def safe_load_model(model_path):
    """安全加载模型"""
    # 使用try-except处理可能的异常
    try:
        # 检查模型文件是否存在
        if Path(model_path).exists():
            # 使用joblib加载模型
            model = joblib.load(model_path)
            # 返回模型和成功信息
            return model, "Model loaded successfully"
        else:
            # 文件不存在时返回None和错误信息
            return None, f"Model file not found: {model_path}"
    # 捕获加载过程中的异常
    except Exception as e:
        # 返回None和错误信息
        return None, f"Error loading model: {str(e)}"

# 定义安全转换为标量的函数
def safe_convert_to_scalar(value):
    """安全地将值转换为Python标量"""
    # 使用try-except处理转换异常
    try:
        # 检查值是否有item方法（numpy标量）
        if hasattr(value, 'item'):
            # 使用item()方法转换为Python标量
            return value.item()
        # 检查值是否有长度且长度为1
        elif hasattr(value, '__len__') and len(value) == 1:
            # 转换为浮点数
            return float(value[0])
        # 检查值是否为长度为1的列表或元组
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            # 转换为浮点数
            return float(value[0])
        else:
            # 直接转换为浮点数
            return float(value)
    # 捕获转换异常
    except (ValueError, TypeError, IndexError):
        # 如果是标量则转换为浮点数，否则返回0.0
        return float(value) if np.isscalar(value) else 0.0

# 定义安全格式化数值的函数
def format_value_safe(value):
    """安全格式化数值，避免numpy格式化错误"""
    # 使用try-except处理格式化异常
    try:
        # 首先安全转换为标量
        scalar_value = safe_convert_to_scalar(value)
        
        # 根据数值大小选择不同的格式化精度
        if abs(scalar_value) < 0.01:
            # 小于0.01的数值保留4位小数
            return f"{scalar_value:.4f}"
        elif abs(scalar_value) < 1:
            # 小于1的数值保留3位小数
            return f"{scalar_value:.3f}"
        elif abs(scalar_value) < 100:
            # 小于100的数值保留2位小数
            return f"{scalar_value:.2f}"
        else:
            # 大于等于100的数值保留1位小数
            return f"{scalar_value:.1f}"
    # 捕获格式化异常
    except Exception:
        # 异常时直接转换为字符串
        return str(value)

# 注释：加载背景数据

# 从CSV文件读取糖尿病数据集，并选择指定的特征列作为背景数据
background_data_raw = pd.read_csv("diabetes.csv").loc[:,["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]

# %%


# 定义UI
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            .card { 
                border-radius: 10px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
                margin-bottom: 20px;
            }
            .btn-primary { 
                background-color: #007bff; 
                border-color: #007bff;
                border-radius: 5px;
            }
            .form-control {
                border-radius: 5px;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
        """)
    ),
    
    ui.div(
        ui.h1("糖尿病预测模型 - SHAP分析", 
               style="text-align: center; color: #2c3e50; margin-bottom: 30px; font-weight: bold;"),
        style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; margin: -15px -15px 20px -15px;"
    ),
    
    ui.row(
        ui.column(4,
            ui.div(
                ui.h3("模型状态", style="color: #2c3e50; margin-bottom: 15px;"),
                ui.output_ui("model_status"),
                class_="card",
                style="padding: 20px;"
            ),
            
            ui.div(
                ui.h3("特征输入", style="color: #2c3e50; margin-bottom: 15px;"),
                ui.output_ui("feature_inputs"),
                ui.br(),
                ui.input_action_button("predict", "开始预测", 
                                     class_="btn btn-primary btn-lg",
                                     style="width: 100%; margin-top: 10px;"),
                class_="card",
                style="padding: 20px;"
            )
        ),
        
        ui.column(8,
            ui.div(
                ui.h3("预测结果", style="color: #2c3e50; margin-bottom: 15px;"),
                ui.output_ui("prediction_result"),
                class_="card",
                style="padding: 20px; min-height: 200px;"
            ),
            
            ui.div(
                ui.h3("SHAP分析", style="color: #2c3e50; margin-bottom: 15px;"),
                ui.input_radio_buttons(
                    "plot_type",
                    "选择图表类型:",
                    choices={
                        "shap_bar": "SHAP条形图",
                        "waterfall": "瀑布图", 
                        "original_force": "力图",
                        "custom_bar": "自定义条形图"
                    },
                    selected="shap_bar",
                    inline=True
                ),
                ui.output_ui("shap_plot"),
                class_="card",
                style="padding: 20px;"
            )
        )
    )
)

# 定义服务器逻辑
def server(input, output, session):
    
    @output
    @render.ui
    def model_status():
        """显示模型状态"""
        model_name = "diabetes_rf"
        config = MODEL_CONFIG[model_name]
        model, status = safe_load_model(config["file"])
        
        if model is not None:
            status_color = "#28a745"  # 绿色
            status_icon = "✓"
        else:
            status_color = "#dc3545"  # 红色
            status_icon = "✗"
        
        return ui.div(
            ui.h4(f"{status_icon} 糖尿病随机森林模型", 
                  style=f"color: {status_color}; margin-bottom: 10px;"),
            ui.p(f"状态: {status}", style="color: #666; margin-bottom: 5px;"),
            ui.p(f"特征数量: {len(config['features'])}", style="color: #666;"),
            style="border-left: 4px solid " + status_color + "; padding-left: 15px;"
        )
    
    @output
    @render.ui  
    def feature_inputs():
        """动态生成特征输入字段"""
        config = MODEL_CONFIG["diabetes_rf"]
        features = config["features"]
        
        # 为diabetes数据集设置合理的默认值
        default_values = {
            "Pregnancies": 3,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 25.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 30
        }
        
        inputs = []
        for feature in features:
            feature_id = sanitize_id(feature)
            default_val = default_values.get(feature, 0)
            
            # 根据特征类型设置不同的输入控件
            if feature in ["Pregnancies", "Age"]:
                inputs.append(
                    ui.input_numeric(feature_id, f"{feature}:", 
                                   value=default_val, min=0, max=20 if feature == "Pregnancies" else 120)
                )
            elif feature == "Glucose":
                inputs.append(
                    ui.input_numeric(feature_id, f"{feature}:", 
                                   value=default_val, min=0, max=300)
                )
            elif feature in ["BloodPressure", "SkinThickness"]:
                inputs.append(
                    ui.input_numeric(feature_id, f"{feature}:", 
                                   value=default_val, min=0, max=200)
                )
            elif feature == "Insulin":
                inputs.append(
                    ui.input_numeric(feature_id, f"{feature}:", 
                                   value=default_val, min=0, max=900)
                )
            elif feature == "BMI":
                inputs.append(
                    ui.input_numeric(feature_id, f"{feature}:", 
                                   value=default_val, min=10.0, max=70.0, step=0.1)
                )
            elif feature == "DiabetesPedigreeFunction":
                inputs.append(
                    ui.input_numeric(feature_id, f"{feature}:", 
                                   value=default_val, min=0.0, max=3.0, step=0.01)
                )
            else:
                inputs.append(
                    ui.input_numeric(feature_id, f"{feature}:", value=default_val)
                )
        
        return ui.div(*inputs)
    
    @reactive.Calc
    def load_model():
        """加载模型"""
        config = MODEL_CONFIG["diabetes_rf"]
        model, status = safe_load_model(config["file"])
        return model, config["features"], status
    
    
    
    @output
    @render.ui
    def prediction_result():
        """显示预测结果"""
        if input.predict() == 0:
            return ui.div(
                ui.p("请点击'开始预测'按钮进行预测", style="color: #666; text-align: center; margin-top: 50px;")
            )
        
        model, features, status = load_model()
        
        if model is None:
            return ui.div(
                ui.p(f"模型加载失败: {status}", style="color: red;"),
                style="background-color: #f8d7da; padding: 10px; border-radius: 5px;"
            )
        
        try:
            # 收集输入数据
            input_data = []
            for feature in features:
                feature_id = sanitize_id(feature)
                value = getattr(input, feature_id)()
                input_data.append(value)
            
            # 创建DataFrame
            input_df = pd.DataFrame([input_data], columns=features)
            
            # 进行预测
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # 格式化结果
            result_text = "糖尿病" if prediction == 1 else "非糖尿病"
            confidence = max(prediction_proba) * 100
            
            # 设置结果颜色
            if prediction == 1:
                result_color = "#dc3545"  # 红色
                bg_color = "#f8d7da"
            else:
                result_color = "#28a745"  # 绿色
                bg_color = "#d4edda"
            
            return ui.div(
                ui.h4(f"预测结果: {result_text}", 
                      style=f"color: {result_color}; margin-bottom: 15px; font-weight: bold;"),
                ui.p(f"置信度: {confidence:.2f}%", style="color: #333; font-size: 16px;"),
                ui.p(f"非糖尿病概率: {prediction_proba[0]:.4f}", style="color: #666;"),
                ui.p(f"糖尿病概率: {prediction_proba[1]:.4f}", style="color: #666;"),
                ui.hr(),
                ui.h5("输入特征值:", style="color: #333; margin-bottom: 10px;"),
                ui.div(
                    *[ui.p(f"{feature}: {value}", style="color: #666; margin: 2px 0;") 
                      for feature, value in zip(features, input_data)],
                    style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;"
                ),
                style=f"background-color: {bg_color}; padding: 20px; border-radius: 8px; border-left: 4px solid {result_color};"
            )
            
        except Exception as e:
            return ui.div(
                ui.p(f"预测过程中出现错误: {str(e)}", style="color: red;"),
                style="background-color: #f8d7da; padding: 10px; border-radius: 5px;"
            )
    
    
    
    @output
    @render.ui
    def shap_plot():
        """生成SHAP图"""
        if input.predict() == 0:
            return ui.div(
                ui.p("请先进行预测以查看SHAP分析", style="color: #666; text-align: center; margin-top: 50px;")
            )
        
        model, features, status = load_model()
        
        if model is None:
            return ui.div(
                ui.p(f"模型加载失败: {status}", style="color: red;"),
                style="background-color: #f8d7da; padding: 10px; border-radius: 5px;"
            )
        
        try:
            # 收集输入数据
            input_data = []
            for feature in features:
                feature_id = sanitize_id(feature)
                value = getattr(input, feature_id)()
                input_data.append(float(value))  # 确保转换为Python float
            
            input_df = pd.DataFrame([input_data], columns=features)
            print('输入数据：')
            print(input_df)
            # 获取背景数据
            global background_data_raw
            background_data = background_data_raw.copy() 
            print('背景数据：')
            print(background_data)

            n_background = min(100, len(background_data))
            if len(background_data) > n_background:
                background_data = background_data.sample(n=n_background, random_state=42)

            background_data = background_data.reset_index(drop=True)
            

            # 计算SHAP值
            explainer = None
            shap_values = None
            expected_value = None
            
            try:
                if str(type(model)).lower().find('xgb') != -1 or hasattr(model, 'get_booster'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_df)
                    expected_value = explainer.expected_value
                elif hasattr(model, 'tree_') or hasattr(model, 'estimators_') or str(type(model)).lower().find('forest') != -1:
                    explainer = shap.TreeExplainer(model, background_data)
                    shap_values = explainer.shap_values(input_df)
                    expected_value = explainer.expected_value
                    print('树模型成功')
                elif hasattr(model, 'coef_'):
                    explainer = shap.LinearExplainer(model, background_data)
                    shap_values = explainer.shap_values(input_df)
                    expected_value = explainer.expected_value
                else:
                    if hasattr(model, 'predict_proba'):
                        predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=features))[:, 1]
                    else:
                        predict_fn = lambda x: model.predict(pd.DataFrame(x, columns=features))
                    
                    explainer = shap.KernelExplainer(predict_fn, background_data.values, nsamples=100)
                    shap_values = explainer.shap_values(input_df.values)
                    expected_value = explainer.expected_value
                    print('kernel模型成功')
                    
            except Exception as e:
                print(f"TreeExplainer failed, using KernelExplainer: {e}")
                if hasattr(model, 'predict_proba'):
                    predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=features))[:, 1]
                else:
                    predict_fn = lambda x: model.predict(pd.DataFrame(x, columns=features))
                
                explainer = shap.KernelExplainer(predict_fn, background_data.values, nsamples=50)
                shap_values = explainer.shap_values(input_data.values)
                expected_value = explainer.expected_value

            print(shap_values)
            
            # 处理SHAP值
            if isinstance(shap_values, list):
                if len(shap_values) >= 2:
                    shap_vals = shap_values[1][0] if len(shap_values[1].shape) > 1 else shap_values[1]
                    base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                else:
                    shap_vals = shap_values[0][0] if len(shap_values[0].shape) > 1 else shap_values[0]
                    base_value = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
            else:
                shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                base_value = expected_value
            
            shap_vals = np.array(shap_vals).flatten()
            print('计算base_values')

            base_value = explainer.expected_value
            print(type(base_value))
            if isinstance(base_value, np.ndarray):
                print('base_value是列表')
                base_value = float(base_value[1])
            else:
                print('base_value不是列表')
                base_value = float(base_value)
            print(base_value)
            plot_type = input.plot_type()
            
            # 设置绘图样式
            plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            print('开始绘图')
            if plot_type == "shap_bar":
                # SHAP条形图
                plt.figure(figsize=(10, 6), dpi=120)
                
                try:
                    explanation = shap.Explanation(
                        values=shap_vals,
                        base_values=base_value,
                        data=np.array(input_data, dtype=float),
                        feature_names=features
                    )
                    
                    shap.plots.bar(explanation, show_data=True, show=False)
                    plt.title(f'SHAP特征重要性分析 - 糖尿病预测模型\n(特征重要性及数据值)')
                    plt.tight_layout()
                    
                except Exception:
                    # 回退到自定义条形图
                    plt.close()
                    plt.figure(figsize=(12, 8))
                    
                    indices = np.argsort(np.abs(shap_vals))[::-1]
                    sorted_shap = shap_vals[indices]
                    sorted_features = [features[i] for i in indices]
                    sorted_values = [input_data[i] for i in indices]
                    
                    y_pos = np.arange(len(sorted_features))
                    colors = ['#ff7f7f' if val > 0 else '#7fbfff' for val in sorted_shap]
                    
                    bars = plt.barh(y_pos, sorted_shap, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
                    
                    max_abs_shap = max(abs(sorted_shap)) if len(sorted_shap) > 0 else 1
                    
                    for i, (feature, value, shap_val) in enumerate(zip(sorted_features, sorted_values, sorted_shap)):
                        value_str = format_value_safe(value)
                        feature_name = feature[:15] + "..." if len(feature) > 15 else feature
                        label_text = f"{feature_name} = {value_str}"
                        
                        if shap_val >= 0:
                            plt.text(shap_val + max_abs_shap * 0.02, i, label_text, 
                                    va='center', ha='left', fontsize=10, fontweight='bold')
                        else:
                            plt.text(shap_val - max_abs_shap * 0.02, i, label_text, 
                                    va='center', ha='right', fontsize=10, fontweight='bold')
                    
                    plt.yticks([])
                    plt.xlabel('SHAP值 (对模型输出的影响)', fontsize=12, fontweight='bold')
                    plt.title(f'SHAP条形图 - 糖尿病预测模型\n基准值: {base_value:.4f}', 
                            fontsize=14, fontweight='bold', pad=20)
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
                    plt.grid(True, alpha=0.3, axis='x')
                    
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='#ff7f7f', alpha=0.8, label='增加预测概率'),
                        Patch(facecolor='#7fbfff', alpha=0.8, label='降低预测概率')
                    ]
                    plt.legend(handles=legend_elements, loc='lower right')
            
            elif plot_type == "waterfall":
                # 瀑布图
                plt.figure(figsize=(12, 8), dpi=120)
                
                try:
                    explanation = shap.Explanation(
                        values=shap_vals,
                        base_values=base_value,
                        data=np.array(input_data, dtype=float),
                        feature_names=features
                    )
                    
                    shap.plots.waterfall(explanation, show=False)
                    plt.title(f'SHAP瀑布图 - 糖尿病预测模型')
                    plt.tight_layout()
                    
                except Exception:
                    # 回退到自定义瀑布图
                    plt.close()
                    plt.figure(figsize=(12, 8))
                    
                    indices = np.argsort(np.abs(shap_vals))[::-1]
                    sorted_shap = shap_vals[indices]
                    sorted_features = [features[i] for i in indices]
                    sorted_values = [input_data[i] for i in indices]
                    
                    y_pos = np.arange(len(sorted_features))
                    colors = ['#ff7f7f' if val > 0 else '#7fbfff' for val in sorted_shap]
                    
                    bars = plt.barh(y_pos, sorted_shap, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
                    
                    max_abs_shap = max(abs(sorted_shap)) if len(sorted_shap) > 0 else 1
                    
                    for i, (feature, value, shap_val) in enumerate(zip(sorted_features, sorted_values, sorted_shap)):
                        value_str = format_value_safe(value)
                        feature_name = feature[:15] + "..." if len(feature) > 15 else feature
                        label = f"{feature_name} = {value_str}"
                        
                        text_offset = 0.02 * max_abs_shap
                        if shap_val >= 0:
                            plt.text(shap_val + text_offset, i, label, va='center', ha='left', fontsize=10,
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                        else:
                            plt.text(shap_val - text_offset, i, label, va='center', ha='right', fontsize=10,
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    
                    plt.yticks([])
                    plt.xlabel('SHAP值 (对模型输出的影响)', fontsize=12, fontweight='bold')
                    plt.title(f'SHAP瀑布图 - 糖尿病预测模型\n基准值: {base_value:.4f}', 
                            fontsize=14, fontweight='bold', pad=20)
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
                    plt.grid(True, alpha=0.3, axis='x')
                    
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='#ff7f7f', alpha=0.8, label='增加预测概率'),
                        Patch(facecolor='#7fbfff', alpha=0.8, label='降低预测概率')
                    ]
                    plt.legend(handles=legend_elements, loc='lower right')
                    
                    plt.tight_layout()
            elif plot_type == "original_force":
                    # 原始SHAP力图
                    plt.figure(figsize=(12, 8))
                    
                    # 使用shap.force_plot生成力图数据
                    force_plot_data = shap.force_plot(
                        base_value, shap_vals, input_data, 
                        feature_names=features, matplotlib=True, show=False
                    )
                    
                    plt.title(f'SHAP力图 - 糖尿病预测模型')
                    # plt.tight_layout()
            
            else:  # custom_bar
                # 自定义条形图
                plt.figure(figsize=(12, 8))
                
                indices = np.argsort(np.abs(shap_vals))[::-1]
                sorted_shap = shap_vals[indices]
                sorted_features = [features[i] for i in indices]
                sorted_values = [input_data[i] for i in indices]
                
                y_pos = np.arange(len(sorted_features))
                colors = ['#ff7f7f' if val > 0 else '#7fbfff' for val in sorted_shap]
                
                bars = plt.barh(y_pos, sorted_shap, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
                
                max_abs_shap = max(abs(sorted_shap)) if len(sorted_shap) > 0 else 1
                
                for i, (feature, value, shap_val) in enumerate(zip(sorted_features, sorted_values, sorted_shap)):
                    value_str = format_value_safe(value)
                    feature_name = feature[:15] + "..." if len(feature) > 15 else feature
                    label = f"{feature_name} = {value_str}"
                    
                    text_offset = 0.02 * max_abs_shap
                    if shap_val >= 0:
                        plt.text(shap_val + text_offset, i, label, va='center', ha='left', fontsize=10,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    else:
                        plt.text(shap_val - text_offset, i, label, va='center', ha='right', fontsize=10,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                plt.yticks([])
                plt.xlabel('SHAP值 (对模型输出的影响)', fontsize=12, fontweight='bold')
                plt.title(f'SHAP力图 - 糖尿病预测模型\n基准值: {base_value:.4f}', 
                        fontsize=14, fontweight='bold', pad=20)
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
                plt.grid(True, alpha=0.3, axis='x')
                
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#ff7f7f', alpha=0.8, label='增加预测概率'),
                    Patch(facecolor='#7fbfff', alpha=0.8, label='降低预测概率')
                ]
                plt.legend(handles=legend_elements, loc='lower right')              
                
                plt.tight_layout()
            
            # 保存图形
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # 根据图形类型提供不同的解释
            if plot_type == "shap_bar":
                interpretation_text = [
                    ui.h4("SHAP条形图解释:", style="color: #2c3e50; margin-top: 20px;"),
                    ui.p("• 此图显示每个特征的SHAP值及其实际数据值", style="color: #333;"),
                    ui.p("• 正SHAP值（红色）增加糖尿病预测概率", style="color: #e74c3c;"),
                    ui.p("• 负SHAP值（蓝色）降低糖尿病预测概率", style="color: #007bff;"),
                    ui.p("• 特征按绝对重要性排序", style="color: #333;"),
                    ui.p(f"• 基准值: {base_value:.4f}", style="color: #333;"),
                ]
            elif plot_type == "waterfall":
                interpretation_text = [
                    ui.h4("SHAP瀑布图解释:", style="color: #2c3e50; margin-top: 20px;"),
                    ui.p("• 瀑布图显示每个特征如何影响最终预测", style="color: #333;"),
                    ui.p("• 从基准值开始，每个特征将预测向上或向下推动", style="color: #333;"),
                    ui.p("• 红色条表示增加预测的特征", style="color: #e74c3c;"),
                    ui.p("• 蓝色条表示降低预测的特征", style="color: #007bff;"),
                    ui.p("• 右侧的最终值显示模型的预测结果", style="color: #333;"),
                    ui.p(f"• 基准值: {base_value:.4f}", style="color: #333;"),
                ]
            else:  # original_force 或 custom_bar
                interpretation_text = [
                    ui.h4("SHAP力图解释:", style="color: #2c3e50; margin-top: 20px;"),
                    ui.p("• 力图可视化特征如何将预测从基准值推开", style="color: #333;"),
                    ui.p("• 红色特征将预测推向更高值（正贡献）", style="color: #e74c3c;"),
                    ui.p("• 蓝色特征将预测推向更低值（负贡献）", style="color: #007bff;"),
                    ui.p("• 每个段的宽度代表特征影响的大小", style="color: #333;"),
                    ui.p("• 特征排列显示对预测的累积效应", style="color: #333;"),
                    ui.p(f"• 基准值: {base_value:.4f}", style="color: #333;"),
                ]
            
            # 返回UI
            return ui.div(
                ui.img(src=f"data:image/png;base64,{image_base64}", 
                       style="width: 100%; max-width: 900px; border: 1px solid #ddd; border-radius: 8px;"),
                ui.br(),
                ui.div(
                    *interpretation_text,
                    style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 15px;"
                )
            )
            
        except Exception as e:
            return ui.div(
                ui.p(f"生成SHAP图时出错: {str(e)}", style="color: red;"),
                ui.p("这可能是由于模型兼容性问题或缺少依赖项。", style="color: #666;"),
                style="background-color: #f8d7da; padding: 10px; border-radius: 5px; border-left: 4px solid #dc3545;"
            )

# 创建应用实例
app = App(app_ui, server)

# 运行应用
if __name__ == "__main__":
    app.run(port=8001)


