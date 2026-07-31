import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

DATA_PATH = "telco_churn.csv"

def load_raw_data():
    """Load original raw dataset."""
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess_data(df):
    """
    Clean dataset and prepare X and y matrices using get_dummies(drop_first=True).
    """
    df_prep = df.copy()
    if 'customerID' in df_prep.columns:
        df_prep.drop(columns=['customerID'], inplace=True)
    
    # Handle TotalCharges missing/blank values
    df_prep['TotalCharges'] = pd.to_numeric(df_prep['TotalCharges'], errors='coerce')
    df_prep['TotalCharges'] = df_prep['TotalCharges'].fillna(df_prep['TotalCharges'].median())
    
    # Drop duplicates if any
    df_prep.drop_duplicates(inplace=True)
    
    # Map target
    if 'Churn' in df_prep.columns:
        df_prep['Churn'] = df_prep['Churn'].map({'Yes': 1, 'No': 0})
        y = df_prep['Churn']
        df_features = df_prep.drop(columns=['Churn'])
    else:
        y = None
        df_features = df_prep
        
    df_encoded = pd.get_dummies(df_features, drop_first=True)
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    
    return df_encoded, y, df_prep

def train_and_evaluate():
    """
    Train Random Forest model on 80% train / 20% test split and return metrics, model, and artifacts.
    """
    df_raw = load_raw_data()
    X, y, df_prep = preprocess_data(df_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1-Score': round(f1_score(y_test, y_pred), 4),
        'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return {
        'model': rf,
        'metrics': metrics,
        'cm': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': metrics['ROC-AUC'],
        'report_df': report_df,
        'importance_df': importance_df,
        'feature_names': list(X.columns),
        'df_raw': df_raw,
        'df_prep': df_prep,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

# ==================== VISUALIZATION UTILS ====================

def plot_churn_distribution(df):
    """Donut chart for Churn distribution."""
    churn_counts = df['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Status', 'Jumlah']
    
    fig = px.pie(
        churn_counts,
        names='Status',
        values='Jumlah',
        hole=0.4,
        color='Status',
        color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
        title='<b>Distribusi Customer Churn</b>'
    )
    fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), template='plotly_white')
    return fig

def plot_churn_by_contract(df):
    """Bar chart showing Churn Rate (%) per Contract Type."""
    grouped = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    grouped.columns = ['Contract', 'ChurnRate']
    grouped['ChurnRate'] = grouped['ChurnRate'].round(1)
    
    fig = px.bar(
        grouped,
        x='Contract',
        y='ChurnRate',
        text=grouped['ChurnRate'].apply(lambda x: f"{x}%"),
        color='Contract',
        color_discrete_sequence=['#EF4444', '#3B82F6', '#10B981'],
        title='<b>Tingkat Churn Rate (%) Berdasarkan Jenis Kontrak</b>'
    )
    fig.update_traces(textposition='outside', textfont=dict(size=14, weight='bold'))
    fig.update_layout(
        yaxis_title='Churn Rate (%)',
        xaxis_title='Tipe Kontrak',
        template='plotly_white',
        showlegend=False,
        margin=dict(t=50, b=30, l=30, r=30),
        yaxis=dict(range=[0, max(grouped['ChurnRate']) + 10])
    )
    return fig

def plot_churn_by_tenure_group(df):
    """Bar chart showing Churn Rate (%) across tenure groups."""
    df_temp = df.copy()
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['< 1 Thn', '1-2 Thn', '2-3 Thn', '3-4 Thn', '4-5 Thn', '> 5 Thn']
    df_temp['TenureGroup'] = pd.cut(df_temp['tenure'], bins=bins, labels=labels, include_lowest=True)
    
    grouped = df_temp.groupby('TenureGroup', observed=False)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    grouped.columns = ['TenureGroup', 'ChurnRate']
    grouped['ChurnRate'] = grouped['ChurnRate'].round(1)
    
    fig = px.line(
        grouped,
        x='TenureGroup',
        y='ChurnRate',
        markers=True,
        text=grouped['ChurnRate'].apply(lambda x: f"{x}%"),
        title='<b>Tren Churn Rate (%) Berdasarkan Lama Berlangganan (Tenure)</b>'
    )
    fig.update_traces(
        line=dict(color='#EF4444', width=4),
        marker=dict(size=10, color='#1E293B'),
        textposition='top center',
        textfont=dict(size=12, weight='bold')
    )
    fig.update_layout(
        yaxis_title='Churn Rate (%)',
        xaxis_title='Kategori Masa Berlangganan (Tenure)',
        template='plotly_white',
        margin=dict(t=50, b=30, l=30, r=30),
        yaxis=dict(range=[0, max(grouped['ChurnRate']) + 10])
    )
    return fig

def plot_churn_by_monthly_charges_group(df):
    """Bar chart showing Churn Rate per Monthly Charges Tier."""
    df_temp = df.copy()
    bins = [0, 35, 70, 100, 150]
    labels = ['Sangat Murah (< $35)', 'Sedang ($35-$70)', 'Tinggi ($70-$100)', 'Sangat Tinggi (> $100)']
    df_temp['ChargeTier'] = pd.cut(df_temp['MonthlyCharges'], bins=bins, labels=labels, include_lowest=True)
    
    grouped = df_temp.groupby('ChargeTier', observed=False)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    grouped.columns = ['ChargeTier', 'ChurnRate']
    grouped['ChurnRate'] = grouped['ChurnRate'].round(1)
    
    fig = px.bar(
        grouped,
        x='ChargeTier',
        y='ChurnRate',
        text=grouped['ChurnRate'].apply(lambda x: f"{x}%"),
        color='ChurnRate',
        color_continuous_scale='Reds',
        title='<b>Tingkat Churn Rate (%) Berdasarkan Kategori Biaya Bulanan</b>'
    )
    fig.update_traces(textposition='outside', textfont=dict(size=13, weight='bold'))
    fig.update_layout(
        yaxis_title='Churn Rate (%)',
        xaxis_title='Kategori Biaya Bulanan',
        template='plotly_white',
        coloraxis_showscale=False,
        margin=dict(t=50, b=30, l=30, r=30),
        yaxis=dict(range=[0, max(grouped['ChurnRate']) + 10])
    )
    return fig

def plot_churn_by_payment_method(df):
    """Horizontal Bar chart showing Churn Rate by Payment Method."""
    grouped = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    grouped.columns = ['PaymentMethod', 'ChurnRate']
    grouped['ChurnRate'] = grouped['ChurnRate'].round(1)
    grouped = grouped.sort_values(by='ChurnRate', ascending=True)
    
    fig = px.bar(
        grouped,
        x='ChurnRate',
        y='PaymentMethod',
        orientation='h',
        text=grouped['ChurnRate'].apply(lambda x: f"{x}%"),
        color='ChurnRate',
        color_continuous_scale='Oranges',
        title='<b>Risiko Churn Rate (%) Per Metode Pembayaran</b>'
    )
    fig.update_traces(textposition='outside', textfont=dict(size=12, weight='bold'))
    fig.update_layout(
        xaxis_title='Churn Rate (%)',
        yaxis_title='Metode Pembayaran',
        template='plotly_white',
        coloraxis_showscale=False,
        margin=dict(t=50, b=30, l=30, r=30),
        xaxis=dict(range=[0, max(grouped['ChurnRate']) + 10])
    )
    return fig

def plot_services_impact(df):
    """Bar chart comparing Churn Rate for key value-added services."""
    services = ['TechSupport', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection']
    records = []
    for s in services:
        sub = df[df[s].isin(['Yes', 'No'])]
        for status in ['No', 'Yes']:
            sub_st = sub[sub[s] == status]
            rate = (sub_st['Churn'] == 'Yes').mean() * 100 if len(sub_st) > 0 else 0
            records.append({
                'Layanan': s,
                'StatusLayanan': 'Punya Layanan' if status == 'Yes' else 'Tidak Punya',
                'ChurnRate': round(rate, 1)
            })
    res_df = pd.DataFrame(records)
    
    fig = px.bar(
        res_df,
        x='Layanan',
        y='ChurnRate',
        color='StatusLayanan',
        barmode='group',
        text=res_df['ChurnRate'].apply(lambda x: f"{x}%"),
        color_discrete_map={'Punya Layanan': '#10B981', 'Tidak Punya': '#EF4444'},
        title='<b>Perbandingan Churn Rate: Memiliki vs Tidak Memiliki Layanan Tambahan</b>'
    )
    fig.update_traces(textposition='outside', textfont=dict(size=11, weight='bold'))
    fig.update_layout(
        yaxis_title='Churn Rate (%)',
        xaxis_title='Fitur Layanan Tambahan',
        template='plotly_white',
        legend_title='Status Fitur',
        margin=dict(t=50, b=30, l=30, r=30),
        yaxis=dict(range=[0, max(res_df['ChurnRate']) + 12])
    )
    return fig

def plot_internet_service_churn(df):
    """Bar chart for Churn Rate by Internet Service type."""
    grouped = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    grouped.columns = ['InternetService', 'ChurnRate']
    grouped['ChurnRate'] = grouped['ChurnRate'].round(1)
    
    fig = px.bar(
        grouped,
        x='InternetService',
        y='ChurnRate',
        text=grouped['ChurnRate'].apply(lambda x: f"{x}%"),
        color='InternetService',
        color_discrete_sequence=['#3B82F6', '#EF4444', '#10B981'],
        title='<b>Churn Rate (%) Berdasarkan Jenis Layanan Internet</b>'
    )
    fig.update_traces(textposition='outside', textfont=dict(size=13, weight='bold'))
    fig.update_layout(
        yaxis_title='Churn Rate (%)',
        xaxis_title='Penyedia Internet',
        template='plotly_white',
        showlegend=False,
        margin=dict(t=50, b=30, l=30, r=30),
        yaxis=dict(range=[0, max(grouped['ChurnRate']) + 10])
    )
    return fig

def plot_confusion_matrix(cm):
    """Plotly Heatmap for Confusion Matrix."""
    labels = ['No Churn (0)', 'Churn (1)']
    text = [[f'TN: {cm[0,0]}', f'FP: {cm[0,1]}'],
            [f'FN: {cm[1,0]}', f'TP: {cm[1,1]}']]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues',
        showscale=False
    ))
    fig.update_layout(
        title='<b>Confusion Matrix Model Random Forest</b>',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        template='plotly_white',
        margin=dict(t=50, b=30, l=30, r=30)
    )
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plotly line chart for ROC Curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'Random Forest (AUC = {roc_auc:.4f})',
        line=dict(color='#2563EB', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#DC2626', dash='dash', width=2)
    ))
    fig.update_layout(
        title='<b>Receiver Operating Characteristic (ROC) Curve</b>',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        template='plotly_white',
        margin=dict(t=50, b=30, l=30, r=30)
    )
    return fig

def plot_feature_importance(importance_df, top_n=15):
    """Plotly horizontal bar chart for top N Feature Importances."""
    top_df = importance_df.head(top_n).sort_values(by='Importance', ascending=True)
    fig = px.bar(
        top_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'<b>Top {top_n} Feature Importance - Random Forest</b>',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(template='plotly_white', margin=dict(t=50, b=30, l=30, r=30))
    return fig

# ==================== PREDICTION ENGINE ====================

def align_features(raw_input_df, feature_names):
    """
    Transforms user input dataframe (raw features) into encoded feature vector
    matching the trained model's feature names.
    """
    input_df = raw_input_df.copy()
    if 'customerID' in input_df.columns:
        input_df.drop(columns=['customerID'], inplace=True)
    if 'Churn' in input_df.columns:
        input_df.drop(columns=['Churn'], inplace=True)
        
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
    input_df['TotalCharges'] = input_df['TotalCharges'].fillna(input_df['TotalCharges'].median())
    
    encoded_df = pd.get_dummies(input_df, drop_first=True)
    bool_cols = encoded_df.select_dtypes(include='bool').columns
    encoded_df[bool_cols] = encoded_df[bool_cols].astype(int)
    
    # Reindex columns to match model's expected features
    aligned_df = encoded_df.reindex(columns=feature_names, fill_value=0)
    return aligned_df

def predict_single_customer(model, feature_names, input_data):
    """
    Takes input_data dictionary, returns prediction, probability, risk level & recommendations.
    """
    raw_df = pd.DataFrame([input_data])
    aligned_df = align_features(raw_df, feature_names)
    
    prob = model.predict_proba(aligned_df)[0][1]
    pred = "Yes" if prob >= 0.5 else "No"
    
    # Risk categorization
    if prob >= 0.7:
        risk_level = "Sangat Tinggi (High Risk)"
        risk_color = "#EF4444"
    elif prob >= 0.4:
        risk_level = "Sedang (Medium Risk)"
        risk_color = "#F59E0B"
    else:
        risk_level = "Rendah (Low Risk)"
        risk_color = "#10B981"
        
    # Generate tailored retention recommendations
    recommendations = []
    if input_data.get('Contract') == 'Month-to-month':
        recommendations.append("📌 Ditawarkan diskon khusus / insentif upgrade ke Kontrak 1 atau 2 Tahun.")
    if input_data.get('tenure', 0) <= 12:
        recommendations.append("📌 Masuk ke dalam program welcome/onboarding pelanggan baru untuk membangun loyalitas.")
    if input_data.get('MonthlyCharges', 0) > 70:
        recommendations.append("📌 Berikan penawaran paket hemat / bundling kuota agar biaya bulanan terasa lebih terjangkau.")
    if input_data.get('TechSupport') == 'No':
        recommendations.append("📌 Tawarkan uji coba gratis layanan Tech Support selama 3 bulan.")
    if input_data.get('OnlineSecurity') == 'No':
        recommendations.append("📌 Tawarkan add-on Online Security gratis untuk 6 bulan pertama.")
        
    if not recommendations:
        recommendations.append("📌 Pertahankan kualitas layanan & lakukan check-in berkala untuk kepuasan pelanggan.")
        
    return {
        'prediction': pred,
        'probability': prob,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'recommendations': recommendations
    }

def predict_batch(model, feature_names, batch_df):
    """
    Takes batch dataframe, aligns features, predicts churn probability & labels.
    """
    aligned_df = align_features(batch_df, feature_names)
    probs = model.predict_proba(aligned_df)[:, 1]
    preds = ["Yes" if p >= 0.5 else "No" for p in probs]
    
    result_df = batch_df.copy()
    result_df['Probability_Churn'] = np.round(probs, 4)
    result_df['Predicted_Churn'] = preds
    return result_df
