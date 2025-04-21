from fastapi import FastAPI, HTTPException, Depends, Request, Response, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('Agg')  # GUI backend'i devre dışı bırak
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import get_db_connection, find_optimal_eps, plot_clusters
import json
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import os
from typing import Optional
import csv
import uuid

# Güvenlik ayarları
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Kullanıcı modeli
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

# Veritabanı kullanıcıları
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("admin123"),
        "disabled": False,
    }
}

# Token modeli
class Token(BaseModel):
    access_token: str
    token_type: str

# Rapor parametreleri modeli
class ReportParams(BaseModel):
    min_samples: int = 3
    export_format: str = "csv"
    include_visualizations: bool = True
    metrics: List[str] = ["mean", "std", "sum"]

app = FastAPI(
    title="Market Analiz API",
    description="DBSCAN kullanarak ürün, tedarikçi ve ülke segmentasyonu yapan API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Kullanıcı doğrulama fonksiyonları
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user

class MarketAnalyzer:
    def __init__(self):
        self.engine = get_db_connection()
        self.static_dir = "static"
        if not os.path.exists(self.static_dir):
            os.makedirs(self.static_dir)
        
    def save_plotly_figure(self, fig, filename):
        filepath = os.path.join(self.static_dir, filename)
        fig.write_html(filepath)
        return f"/static/{filename}"
    
    def create_plotly_visualization(self, df, x_col, y_col, title, color_col='cluster'):
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=title,
                        template="plotly_white",
                        hover_data=df.columns)
        
        fig.update_layout(
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            title_font=dict(size=24, color='#2c3e50'),
            font=dict(family="Arial", size=12, color='#2c3e50'),
            xaxis=dict(gridcolor='rgba(0, 0, 0, 0.1)'),
            yaxis=dict(gridcolor='rgba(0, 0, 0, 0.1)'),
            legend=dict(
                title_font=dict(size=14),
                font=dict(size=12)
            )
        )
        
        filename = f"scatter_{uuid.uuid4()}.html"
        return self.save_plotly_figure(fig, filename)
    
    def create_3d_visualization(self, df, x_col, y_col, z_col, title):
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                          color='cluster',
                          title=title,
                          template="plotly_white")
        
        fig.update_layout(
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            title_font=dict(size=24, color='#2c3e50'),
            font=dict(family="Arial", size=12, color='#2c3e50')
        )
        
        filename = f"3d_{uuid.uuid4()}.html"
        return self.save_plotly_figure(fig, filename)
    
    def create_box_plot(self, df, value_col, group_col, title):
        fig = px.box(df, x=group_col, y=value_col,
                    color=group_col,
                    title=title,
                    template="plotly_white")
        
        fig.update_layout(
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            title_font=dict(size=24, color='#2c3e50'),
            font=dict(family="Arial", size=12, color='#2c3e50'),
            xaxis=dict(gridcolor='rgba(0, 0, 0, 0.1)'),
            yaxis=dict(gridcolor='rgba(0, 0, 0, 0.1)')
        )
        
        filename = f"box_{uuid.uuid4()}.html"
        return self.save_plotly_figure(fig, filename)

    def find_optimal_eps(self, X_scaled, min_samples=3):
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        
        # KNN hesapla
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)
        
        # Mesafeleri sırala
        distances = np.sort(distances[:, min_samples-1], axis=0)
        
        # Diz bölgesini bul
        from kneed import KneeLocator
        kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
        eps = distances[kneedle.knee]
        
        return eps

    def analyze_products(self, min_samples: int = 3) -> Dict:
        query = """
        SELECT 
            p.product_id,
            p.product_name,
            p.category_id,
            AVG(od.unit_price) as avg_price,
            COUNT(DISTINCT o.order_id) as order_frequency,
            AVG(od.quantity) as avg_quantity_per_order,
            COUNT(DISTINCT o.customer_id) as unique_customers,
            SUM(od.quantity) as total_quantity_sold
        FROM 
            products p
            INNER JOIN order_details od ON p.product_id = od.product_id
            INNER JOIN orders o ON od.order_id = o.order_id
        GROUP BY 
            p.product_id, p.product_name, p.category_id
        """
        df = pd.read_sql_query(query, self.engine)
        
        features = ["avg_price", "order_frequency", "avg_quantity_per_order", "unique_customers"]
        X = df[features]
        X_scaled = StandardScaler().fit_transform(X)
        
        eps = self.find_optimal_eps(X_scaled, min_samples=min_samples)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df['cluster'] = dbscan.fit_predict(X_scaled)
        
        scatter_plot = self.create_plotly_visualization(
            df, 'order_frequency', 'avg_price', 
            'Ürün Segmentasyonu (DBSCAN)'
        )
        
        box_plot = self.create_box_plot(
            df, 'avg_price', 'cluster',
            'Kümeler Bazında Fiyat Dağılımı'
        )
        
        df['scaled_quantity'] = StandardScaler().fit_transform(df[['avg_quantity_per_order']])
        df['scaled_customers'] = StandardScaler().fit_transform(df[['unique_customers']])
        df['scaled_price'] = StandardScaler().fit_transform(df[['avg_price']])
        
        plot_3d = self.create_3d_visualization(
            df, 'scaled_quantity', 'scaled_customers', 'scaled_price',
            '3D Ürün Segmentasyonu'
        )
        
        outliers = df[df['cluster'] == -1]
        
        # Küme istatistiklerini hesapla
        cluster_stats = {}
        for cluster in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster]
            stats = {
                'avg_price': {
                    'mean': cluster_data['avg_price'].mean(),
                    'std': cluster_data['avg_price'].std()
                },
                'order_frequency': {
                    'mean': cluster_data['order_frequency'].mean(),
                    'std': cluster_data['order_frequency'].std()
                },
                'total_quantity_sold': cluster_data['total_quantity_sold'].sum()
            }
            cluster_stats[str(cluster)] = stats
        
        return {
            "total_products": len(df),
            "outlier_count": len(outliers),
            "cluster_distribution": df['cluster'].value_counts().to_dict(),
            "cluster_statistics": cluster_stats,
            "outlier_products": outliers[['product_id', 'product_name', 'avg_price', 'order_frequency']].to_dict('records'),
            "visualizations": {
                "scatter_plot": scatter_plot,
                "box_plot": box_plot,
                "plot_3d": plot_3d
            }
        }
    
    def analyze_suppliers(self, min_samples: int = 3) -> Dict:
        query = """
        SELECT 
            s.supplier_id,
            s.company_name,
            s.country,
            COUNT(DISTINCT p.product_id) as product_count,
            SUM(od.quantity) as total_quantity_sold,
            AVG(od.unit_price) as avg_price,
            COUNT(DISTINCT o.customer_id) as unique_customers,
            SUM(od.unit_price * od.quantity) as total_revenue
        FROM 
            suppliers s
            INNER JOIN products p ON s.supplier_id = p.supplier_id
            INNER JOIN order_details od ON p.product_id = od.product_id
            INNER JOIN orders o ON od.order_id = o.order_id
        GROUP BY 
            s.supplier_id, s.company_name, s.country
        """
        df = pd.read_sql_query(query, self.engine)
        
        features = ["product_count", "total_quantity_sold", "avg_price", "unique_customers"]
        X = df[features]
        X_scaled = StandardScaler().fit_transform(X)
        
        eps = find_optimal_eps(X_scaled, min_samples=min_samples)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df['cluster'] = dbscan.fit_predict(X_scaled)
        
        scatter_plot = self.create_plotly_visualization(
            df, 'total_quantity_sold', 'avg_price',
            'Tedarikçi Segmentasyonu (DBSCAN)'
        )
        
        box_plot = self.create_box_plot(
            df, 'total_revenue', 'cluster',
            'Kümeler Bazında Gelir Dağılımı'
        )
        
        df['scaled_products'] = StandardScaler().fit_transform(df[['product_count']])
        df['scaled_customers'] = StandardScaler().fit_transform(df[['unique_customers']])
        df['scaled_revenue'] = StandardScaler().fit_transform(df[['total_revenue']])
        
        plot_3d = self.create_3d_visualization(
            df, 'scaled_products', 'scaled_customers', 'scaled_revenue',
            '3D Tedarikçi Segmentasyonu'
        )
        
        outliers = df[df['cluster'] == -1]
        
        # Küme istatistiklerini hesapla
        cluster_stats = {}
        for cluster in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster]
            stats = {
                'product_count': {
                    'mean': cluster_data['product_count'].mean(),
                    'std': cluster_data['product_count'].std()
                },
                'total_revenue': cluster_data['total_revenue'].sum(),
                'unique_customers': {
                    'mean': cluster_data['unique_customers'].mean(),
                    'std': cluster_data['unique_customers'].std()
                }
            }
            cluster_stats[str(cluster)] = stats
        
        return {
            "total_suppliers": len(df),
            "outlier_count": len(outliers),
            "cluster_distribution": df['cluster'].value_counts().to_dict(),
            "cluster_statistics": cluster_stats,
            "outlier_suppliers": outliers[['supplier_id', 'company_name', 'country', 'total_revenue']].to_dict('records'),
            "visualizations": {
                "scatter_plot": scatter_plot,
                "box_plot": box_plot,
                "plot_3d": plot_3d
            }
        }
    
    def analyze_countries(self, min_samples: int = 3) -> Dict:
        query = """
        SELECT 
            c.country,
            COUNT(DISTINCT o.order_id) as total_orders,
            AVG(od.unit_price * od.quantity) as avg_order_value,
            AVG(od.quantity) as avg_products_per_order,
            COUNT(DISTINCT o.customer_id) as unique_customers,
            SUM(od.unit_price * od.quantity) as total_revenue
        FROM 
            customers c
            INNER JOIN orders o ON c.customer_id = o.customer_id
            INNER JOIN order_details od ON o.order_id = od.order_id
        GROUP BY 
            c.country
        """
        df = pd.read_sql_query(query, self.engine)
        
        features = ["total_orders", "avg_order_value", "avg_products_per_order"]
        X = df[features]
        X_scaled = StandardScaler().fit_transform(X)
        
        eps = find_optimal_eps(X_scaled, min_samples=min_samples)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df['cluster'] = dbscan.fit_predict(X_scaled)
        
        scatter_plot = self.create_plotly_visualization(
            df, 'total_orders', 'avg_order_value',
            'Ülke Segmentasyonu (DBSCAN)'
        )
        
        box_plot = self.create_box_plot(
            df, 'total_revenue', 'cluster',
            'Kümeler Bazında Ülke Gelirleri'
        )
        
        df['scaled_orders'] = StandardScaler().fit_transform(df[['total_orders']])
        df['scaled_products'] = StandardScaler().fit_transform(df[['avg_products_per_order']])
        df['scaled_revenue'] = StandardScaler().fit_transform(df[['total_revenue']])
        
        plot_3d = self.create_3d_visualization(
            df, 'scaled_orders', 'scaled_products', 'scaled_revenue',
            '3D Ülke Segmentasyonu'
        )
        
        outliers = df[df['cluster'] == -1]
        
        # Küme istatistiklerini hesapla
        cluster_stats = {}
        for cluster in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster]
            stats = {
                'total_orders': {
                    'mean': cluster_data['total_orders'].mean(),
                    'std': cluster_data['total_orders'].std()
                },
                'total_revenue': cluster_data['total_revenue'].sum(),
                'unique_customers': {
                    'mean': cluster_data['unique_customers'].mean(),
                    'std': cluster_data['unique_customers'].std()
                }
            }
            cluster_stats[str(cluster)] = stats
        
        return {
            "total_countries": len(df),
            "outlier_count": len(outliers),
            "cluster_distribution": df['cluster'].value_counts().to_dict(),
            "cluster_statistics": cluster_stats,
            "outlier_countries": outliers[['country', 'total_orders', 'total_revenue']].to_dict('records'),
            "visualizations": {
                "scatter_plot": scatter_plot,
                "box_plot": box_plot,
                "plot_3d": plot_3d
            }
        }

    def export_data(self, data: Dict, format: str = "csv", filename: str = "analysis"):
        if format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Metric", "Value"])
            for key, value in data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        writer.writerow([f"{key}_{sub_key}", sub_value])
                else:
                    writer.writerow([key, value])
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}.csv"}
            )
        elif format == "json":
            return JSONResponse(
                content=data,
                headers={"Content-Disposition": f"attachment; filename={filename}.json"}
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

analyzer = MarketAnalyzer()

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analyze/products", response_class=HTMLResponse)
async def analyze_products_page(request: Request, min_samples: int = Query(3, description="Minimum örnek sayısı")):
    try:
        result = analyzer.analyze_products(min_samples)
        return templates.TemplateResponse(
            "analysis_results.html",
            {
                "request": request,
                "title": "Ürün Analizi",
                "result": result
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/suppliers", response_class=HTMLResponse)
async def analyze_suppliers_page(request: Request, min_samples: int = Query(3, description="Minimum örnek sayısı")):
    try:
        result = analyzer.analyze_suppliers(min_samples)
        return templates.TemplateResponse(
            "analysis_results.html",
            {
                "request": request,
                "title": "Tedarikçi Analizi",
                "result": result
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/countries", response_class=HTMLResponse)
async def analyze_countries_page(request: Request, min_samples: int = Query(3, description="Minimum örnek sayısı")):
    try:
        result = analyzer.analyze_countries(min_samples)
        return templates.TemplateResponse(
            "analysis_results.html",
            {
                "request": request,
                "title": "Ülke Analizi",
                "result": result
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/{analysis_type}")
async def export_analysis(
    analysis_type: str,
    params: ReportParams,
    current_user: User = Depends(get_current_user)
):
    try:
        if analysis_type == "products":
            data = analyzer.analyze_products(params.min_samples)
        elif analysis_type == "suppliers":
            data = analyzer.analyze_suppliers(params.min_samples)
        elif analysis_type == "countries":
            data = analyzer.analyze_countries(params.min_samples)
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis type")
        
        return analyzer.export_data(
            data,
            format=params.export_format,
            filename=f"{analysis_type}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 