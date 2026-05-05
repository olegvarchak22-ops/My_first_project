import os
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr
from scipy.stats import kruskal
from scipy.stats import median_test
from scipy.stats import kendalltau
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



def download_document(file_name, document_url):
    if os.path.exists(file_name):
        print('The file already exists. Continuing execution.')
        pass
    else:
        response = requests.get(document_url)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
        else:
            print(f'Failed to download the document. Status code: {response.status_code}')

# Налаштування, щоб Pandas показував всі стовпці при виводі
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

file_name = 'ikea.csv'
document_url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv'
download_document(file_name, document_url)

df = pd.read_csv('ikea.csv')


# Інформація про дані у файлі (дивимося на самі дані, чи все там є)
print(f"Розмірність: {df.shape}")

print(f"\nПоглянути на перші 5 рядків: \n{df.head()}")
print(f"\nПоглянути структуру і типи даних:")
print(df.info())
print(f"\nКількість пропусків: \n{df.isna().sum()}")
# Висновок: є пропуски в колонках з розмірами, потрібно буде їх заповнити. Потрібно також буде видалити
# колонки Unnamed, link

# Статистичні Аномалії
print(df.describe().round(2))
print("\nУнікальні записи")
print(df.nunique())

# Перевіряємо на дублікати колонку item_id
dup_count = df.duplicated('item_id').sum()
print(f"\nКількість повних дублікатів: {dup_count}")
df = df.drop_duplicates(subset=['item_id'], keep='first').reset_index(drop=True).copy()
print(f"\nРозмір після видалення дублікатів: {df.shape}")

# Прибираємо порядковий номер і колонку link
df = df.drop(columns=['Unnamed: 0', 'link'])
print(f"\nПоглянути структуру і типи даних:")
print(df.info())

# Заповнення розмірів Медіаною
depth_median = df['depth'].median()
df['depth'] = df['depth'].fillna(depth_median)
height_median = df['height'].median()
df['height'] = df['height'].fillna(height_median)
width_median = df['width'].median()
df['width'] = df['width'].fillna(width_median)

# Перевірка
print("Кількість пропусків після обробки:")
print(df[['depth', 'height', 'width']].isna().sum())

# Попрацюємо з колонкою old_price
# Замінимо значенна No old price NaN
df['old_price'] = df['old_price'].replace('No old price', np.nan)

# Видаляємо текстові символи, а також всі символи з рядка, крім цифр і крапок, щоб перетворити потім стовпець
# old price у числовий тип
df['old_price'] = df['old_price'].str.replace(r'[^\d.]', '', regex=True)
df['old_price'] = df['old_price'].astype(float)

print(df[['old_price', 'price']])
print(df.info())
# В колонці old price є 574 числові значення і стара ціна вища ніж в ціні price , тому припускаємо що на 574 товари
# змінилася ціна в нижчу сторону, можливо це акційна ціна або розпродаж якихось старих моделей

# Обробка колонки designer
print(f"\nКолонка дизайнер: \n{df['designer'].describe()}")
print(df['designer'].value_counts())
print(f"\n Унікальні записи: {df['designer'].nunique()}")
# При аналізі даних, унікальних записів 381, дизайнери від (IKEA of Sweden є 683записи), а також є рядки в яких є записи наприклад
# (IKEA of Sweden/Ola Wihlborg/Ehlén Johansson/Ebba Strandmark)
# приймаю рішення видалити в таких рядках все крім IKEA of Sweden
# Для цього використовую:
# mask — знаходить всі рядки, де є "IKEA of Sweden"
#  .loc[...] = — замінює весь рядок на "IKEA of Sweden"
df = df.copy()
mask = df['designer'].str.contains('IKEA of Sweden', na=False)
df.loc[mask, 'designer'] = 'IKEA of Sweden'

# print(df['designer'].value_counts().to_markdown())
# Після такої очистки команда "IKEA of Sweden" стало 1273 записи, що значно вище за 683, але є ще дуже багато
# рядків де початок з цифр, якогось коду вирішую їх замінити на  "Unknown"
mask = df['designer'].str.contains(r'^\d{3}\.\d{3}\.\d{2}', na=False)
df.loc[mask, 'designer'] = 'Unknown'
print(df['designer'].value_counts().to_markdown())
print(f"\n Унікальні записи: {df['designer'].nunique()}")
# Після цього етапу чистки унікальних записів залишилося 156. На цьому чистку завершено. Переходжу до візуалізацій
#
# Візуалізація по категоріям товарів
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
#
# 1.1 Кількість товарів по категоріям
plt.figure(figsize=(12, 8))
count_category = (df.groupby('category').size().reset_index(name='count'))
ax = sns.barplot(
    x=count_category['category'],
    y=count_category['count'],
    hue = count_category['category'],
    palette='viridis', legend=False
)

# підписи значень
for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.xticks(rotation=45, ha='right')
plt.title('Кількість товарів по категоріях')
plt.xlabel('Категорія')
plt.ylabel('Кількість товарів')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "category_count.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'category_count.png'}")

# 1.2 Медіанна ціна по категоріям
plt.figure(figsize=(12, 8))
median_category = (df.groupby('category')['price'].median().reset_index(name='median'))
ax = sns.barplot(
    x=median_category['category'],
    y=median_category['median'],
    hue = median_category['category'],
    palette='viridis', legend=False
)

# підписи значень
for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.xticks(rotation=45, ha='right')
plt.title('Медіанна ціна по категоріях')
plt.xlabel('Категорія')
plt.ylabel('Медіанна ціна')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "category_median.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'category_median.png'}")

# 1.3 Розкид цін по Скриньковій діаграмі
plt.figure(figsize=(12, 8))
sns.boxplot(x='category', y='price', hue='category', data=df, palette='rocket')
plt.xticks(rotation=45, ha='right')
plt.title('Аналіз цін по категоріям')
plt.xlabel('Категорія')
plt.ylabel('Ціновий діапазон')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "category_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'category_boxplot.png'}")

# 1.4 Розподіл цін
sns.histplot(df['price'], kde=True)
plt.title('Аналіз розподілу цін')
plt.xlabel('Ціна')
plt.ylabel('Кількість')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "count_price.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'count_price.png'}")


# З цього можно зробити висновки, що найбільша кількість представлених товарів в групах Книжкові шафи та стелажі, потім
# стільці, дивани та крісла і категорія стільці та столи
#  Найвища медіанна ціна в категорії шафи, дивани та крісла і ліжка.
# ********************************************************


# 2. Кореляційна матриця. Залежність ціни від розмірів
numeric_df = ['depth', 'height', 'width', 'price']
numeric_data = df[numeric_df]
corr_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Матриця Кореляції між розмірами та ціною')
plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'correlation_heatmap.png'}")

# Найбільша залежнисть ціни від ширини, потім глибини

# **********************************************************
# 3. Топ-15 дизайнерів за кількістю товарів.
top_15_design = (df['designer'].value_counts().head(15).reset_index())
top_15_design.columns = ['designer', 'item_id']
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x=top_15_design['designer'],
    y=top_15_design['item_id'],
    hue = top_15_design['designer'],
    palette='viridis', legend=False
)

# підписи значень
for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.xticks(rotation=45, ha='right')
plt.title('ТОП 15 Дизайнерів за кількістю товарів')
plt.xlabel('Дизайнер')
plt.ylabel('Кількість товарів')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "design_top15.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'design_top15.png'}")

# 3.2 ТОП 15 дизайнерів за кількістю товарів і їх медіанна ціна
top_15 = df['designer'].value_counts().head(15).index
top_designer_median_price = (
    df[df['designer'].isin(top_15)]
      .groupby('designer')['price']
      .median()
      .reset_index(name='median')
      .sort_values('median', ascending=False)
)

plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x=top_designer_median_price['designer'],
    y=top_designer_median_price['median'],
    hue = top_designer_median_price['designer'],
    palette='viridis', legend=False
)

# підписи значень
for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.xticks(rotation=45, ha='right')
plt.title('Медіанна ціна по ТОП 15 Дизайнерам')
plt.xlabel('Дизайнер')
plt.ylabel('Медіанна ціна')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "TOP_15_designer_median_price.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'TOP_15_designer_median_price.png'}")

# 3.3 15 дизайнерів та та їхня медіанна ціна
designer_median_price = (df.groupby('designer')['price'].median().reset_index(name='median').
                         sort_values('median', ascending=False).head(15))
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x=designer_median_price['designer'],
    y=designer_median_price['median'],
    hue = designer_median_price['designer'],
    palette='viridis', legend=False
)
# підписи значень
for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.xticks(rotation=45, ha='right')
plt.title('Медіанна ціна по Дизайнерам')
plt.xlabel('Дизайнер')
plt.ylabel('Медіанна ціна')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "designer_median_price.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'designer_median_price.png'}")

# Найбільше представлено товарів від дизайнерів Ikea, найвища медіанна ціна серед ТОП 15 дизайнерів по кількості
# це K Malmvall/E Lilja Löwenhielm (в них 55 товарів). А ось найвища медіанна ціна по всим дизайнерам це у
# Ehlén Johansson/Fredriksson/L Löwenhielm/Hilland ця група з дизайнерів представлена 1 позицією

# ************************************************************************
# 4. Кольори: Аналіз товарів з/без варіацій кольорів (кількість, ціна)
# 4.1  Рахуємо частки товарів з варіацією кольорів
color_count = df['other_colors'].value_counts()
plt.figure(figsize=(12, 8))
plt.pie(color_count, labels=color_count.index, autopct='%1.0f%%', startangle=90, colors=sns.color_palette('crest'),
        explode=(0.05, 0)
)
plt.title('Розподіл наявності інших кольорів', fontsize=15)
plt.savefig(PLOTS_DIR / "color_count.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'color_count.png'}")
print("\nКількість товарів, які мають інші кольори:", color_count.get('Yes', 0))
print("Кількість товарів, які не мають інших кольорів:", color_count.get('No', 0))

# 4.2 Медіанна ціна на наявність інших кольорів
median_by_colors = df.groupby('other_colors')['price'].median()
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=median_by_colors.index, y=median_by_colors.values, hue=median_by_colors.index,
        palette='viridis', legend=False)

# підписи значень
for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.title('Медіанна ціна на наявність інших кольорів')
plt.xlabel('Наявність інших кольорів')
plt.ylabel('Медіанна ціна')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "color_median_price.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'color_median_price.png'}")
print(f"\nМедіанні ціни: \n{median_by_colors}")

# Наявність інших кольорів є в 45% товарів і медіанна ціна вища там де є інші кольори

# ***********************************************************

# 5. Аналіз товарів, доступних/недоступних онлайн
online_sale = df['sellable_online'].value_counts()
plt.figure(figsize=(12, 8))
plt.pie(online_sale, labels=online_sale.index, autopct='%1.0f%%', startangle=90, colors=sns.color_palette('crest'),
        explode=(0.05, 0)
)
plt.title('Частка продажів онлайн', fontsize=15)
plt.savefig(PLOTS_DIR / "sale_online.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'sale_online.png'}")
print("\nКількість товарів, які продаються онлайн:", online_sale.get(True, 0))
print("Кількість товарів, які продаються тільки в магазинах:", online_sale.get(False, 0))

# Доступність товарів які продаються в онлайн це 99%
# ********************************************************************
# 5.2 Медіанна ціна на товари в продажах онлайн і тільки в магазині
median_by_online = df.groupby('sellable_online')['price'].median()
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=median_by_online.index, y=median_by_online.values, hue=median_by_online.index,
        palette='viridis', legend=False)

# підписи значень
for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.title('Медіанна ціна на наявність/ не наявність товарів онлайн ')
plt.xlabel('Наявність онлайн')
plt.ylabel('Медіанна ціна')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "median_online.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'median_online.png'}")
print(f"\nМедіанні ціни: \n{median_by_online}")

# Доступність товарів які продаються в онлайн це 99% і медіанна ціна вища в онлайн представленності
# ********************************************************************

# 6. Розподіл цін, співвідношення old_price vs price (з лінією регресії)
plt.figure(figsize=(8, 6))
sns.regplot(
    data=df,
    x='old_price',
    y='price'
)
plt.title('Співвідношення старої та нової ціни')
plt.xlabel('Стара ціна')
plt.ylabel('Нова ціна')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "price_regression.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'price_regression.png'}")

# З лінії регресії можна прийти до висновку, що Нова ціна залежить від старої ціни.

#Частина 3: Перевірка Гіпотез

# Гіпотеза 1
# Команда IKEA of Sweden представляє найбільш популярні категорії
# Нульова гіпотеза: IKEA of Sweden не відрізняється від інших дизайнерів за популярністю категорій
# Альтернативна гіпотеза: IKEA of Sweden працює з більш популярними категоріями
# популярність категорій (кількість товарів у категорії)

# Кількість товарів у категоріях
category_counts = df['category'].value_counts()

df = df.copy()
df.loc[:, 'category_popularity'] = df['category'].map(category_counts)

# групи категорій і дизайнерів
ikea = df[df['designer'] == 'IKEA of Sweden']['category_popularity']
others = df[df['designer'] != 'IKEA of Sweden']['category_popularity']
# 1. ANOVA
f_stat, p = f_oneway(ikea, others)
print("\nANOVA p-value:", p)
if p < 0.05:
    print("Нульову гіпотезу відхилено!")
else:
    print("Нульову гіпотезу не відхилено!")
print("Середня ціна:")
print("IKEA of Sweden:", ikea.mean())
print("Others:", others.mean())

# 2. Краскела–Уолліса
h_stat, p = kruskal(ikea, others)
print("\nKruskal-Wallis p-value:", p)
if p < 0.05:
    print("Нульову гіпотезу відхилено!")
else:
    print("Нульову гіпотезу не відхилено!")

# 3. Mann–Whitney U (порівняння медіан, ненормальний розподіл)
u_stat, p = mannwhitneyu(ikea, others, alternative='greater')
print("\nMann-Whitney p-value:", p)
if p < 0.05:
    print("Нульову гіпотезу відхилено!")
else:
    print("Нульову гіпотезу не відхилено!")
print("Медіанна ціна:")
print("IKEA of Sweden:", ikea.median())
print("Others:", others.median())

print("\nГіпотеза 1 Висновок. Гіпотеза не підтверджується. ANOVA (p = 0.798), тест Краскела–Уолліса (p = 0.638), "
      "тест Манна–Уітні (p = 0.319). Команда IKEA of Sweden не представляє найбільш популярні категорії та статистично "
      "не відрізняється від інших дизайнерів за цим показником.")
# Гіпотеза 1. Висновок. Гіпотеза не підтверджується.
# ANOVA (p = 0.798), тест Краскела–Уолліса (p = 0.638), тест Манна–Уітні (p = 0.319)
# Команда IKEA of Sweden не представляє найбільш популярні категорії та статистично
# не відрізняється від інших дизайнерів за цим показником.

# Гіпотеза 2
# Залежність між кількістю товарів в категорії з медіанною ціною
# Нульова гіпотеза : Між кількістю товарів у категорії та медіанною ціною немає значущого зв’язку
# Альтернативна гіпотеза: Між кількістю товарів у категорії та медіанною ціною існує значущий зв’язок
#
# Тест Спірменна
category_stats = (
    df.groupby('category')
      .agg(
          item_count=('item_id', 'count'),
          median_price=('price', 'median')
      )
      .reset_index()
)
corr, p = spearmanr(
    category_stats['item_count'],
    category_stats['median_price']
)
print("\n Тест спірменна p-value:", p)
if p < 0.05:
    print("Нульову гіпотезу відхилено!")
else:
    print("Нульову гіпотезу не відхилено!")
print("Spearman correlation:", corr)

# Тест Кенделла
tau, p = kendalltau(
    category_stats['item_count'],
    category_stats['median_price']
)
print("\n Тест Кенделла p-value:", p)
if p < 0.05:
    print("Нульову гіпотезу відхилено!")
else:
    print("Нульову гіпотезу не відхилено!")
print("Kendall τ:", tau)


print("\n Гіпотеза 2. Висновок по двом тестам. За результатами рангової кореляції Спірмена (ρ = 0.0748, p = 0.7754)"
      "та Кендалла (τ = 0.0517, p = 0.7729) статистично значущого зв’язку між кількістю товарів у категорії "
      "та медіанною ціною не виявлено, але нульову гіпотезу не відхилено тому що 0.77 > 0.05.")

# df = df[df['price'] > 0].copy()
# df['price_ln'] = np.log(df['price'])
#
# category_stats_log = (
#     df.groupby('category')
#       .agg(
#           item_count=('price', 'count'),
#           median_log_price=('price_ln', 'median')
#       )
#       .reset_index()
# )
# corr, p = spearmanr(
#     category_stats_log['item_count'],
#     category_stats_log['median_log_price']
# )
#
# print("Spearman (log):", corr)
# print("p-value:", p)
# # Ідентичні значення отримано як і з не логарифмованою медіанною ціною, тому не включаю це у завдання


# Гіпотеза 2. Висновок по двом тестам. За результатами рангової кореляції Спірмена (ρ = 0.0748, p = 0.7754)
# та Кендалла (τ = 0.0517, p = 0.7729) статистично значущого зв’язку між кількістю товарів у категорії
# та медіанною ціною не виявлено, але нульову гіпотезу не відхилено тому що 0.77 > 0.05.

# Гіпотеза 3.
# Чи є різниця медіанних цін між категоріями
# Нульова гіпотеза : Медіанні ціни товарів одинакові у всих категоріях
# Альтернативна гіпотеза: Медіанні ціни різні по категоріям

# тест Краскела–Уолліса
groups = [group['price'].values for name, group in df.groupby('category')]

stat, p_value = kruskal(*groups)
print("\nТест Краскела–Уолліса p-value:", p_value)
if p_value < 0.05:
    print("Нульову гіпотезу відхилено!")
else:
    print("Нульову гіпотезу не відхилено!")
print("Kruskal-Wallis statistic:", stat)

# Тест Медіан
stat, p, _, _ = median_test(*groups)
print("\n Тест Медіан p-value:", p)
if p_value < 0.05:
    print("Нульову гіпотезу відхилено!")
else:
    print("Нульову гіпотезу не відхилено!")
print("Median test statistic:", stat)

print("\n Гіпотеза 3.  Нульову гіпотезу відхилено так як значення p-value значно меньше 0")
# Гіпотеза 3. Нульову гіпотезу відхилено так як значення p-value значно меньше 0



# ****************************************************************************************************
# Четверта частина ML
# Визначаємо, які колонки використовуємо як фічі
X = df[['category', 'designer', 'other_colors', 'width', 'height', 'depth']]
y = df['price']


# Ділимо на числові і категоріальні ознаки.
num_cols = ['width', 'height', 'depth']
cat_cols = ['category', 'designer', 'other_colors']

# Пайплайн для числових:
# 1) заповнюємо пропуски медіаною, але це зробив попередньо
# 2) масштабуємо StandardScaler (важливо для KNN)
numeric_pipeline = Pipeline(
    steps=[
        # ("imputer" SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
# Пайплайн для категоріальних:
# 1) заповнюємо пропуски модою (most_frequent) або можна "Unknown", пропусків немає
# 2) кодуємо OneHotEncoder (бо порядку немає)
categorical_pipeline = Pipeline(
    steps=[
        # ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# Об’єднуємо все в один препроцесор
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ]
)

# Пробуємо тепер вибрати кращу модель
def getBestRegressor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = [
        LinearRegression(),
        LassoCV(),
        RidgeCV(),
        SVR(kernel="linear"),
        KNeighborsRegressor(n_neighbors=16),
        DecisionTreeRegressor(max_depth=10, random_state=42),
        RandomForestRegressor(random_state=42),
        GradientBoostingRegressor(),
        BaggingRegressor(random_state=42),
        XGBRegressor(objective="reg:squarederror", random_state=42)
    ]

    TestModels = pd.DataFrame()
    res = {}
    tmp = {}

    for model in models:
        model_pipeline = Pipeline(steps=[
            ("prepr", preprocessor),
            ("model", model)
        ])

        m = str(model)
        tmp["Model"] = m[: m.index("(")]
        model_pipeline.fit(X_train, y_train)
        tmp["R^2"] = "{:.5f}".format(model_pipeline.score(X_test, y_test))
        tmp["MAE"] = "{:.5f}".format(
            mean_absolute_error(model_pipeline.predict(X_test), y_test)
        )
        tmp["RMSE"] = "{:.5f}".format(
            np.sqrt(
                mean_squared_error(model_pipeline.predict(X_test), y_test)
            )
        )

        TestModels = pd.concat([TestModels, pd.DataFrame([tmp])])
    TestModels.set_index("Model", inplace=True)
    res["model"] = TestModels
    res["X_train"] = X_train
    res["y_train"] = y_train
    res["X_test"] = X_test
    res["y_test"] = y_test
    return res

# Виклик функції
model = getBestRegressor(X, y)
model_info = model["model"].sort_values(by="R^2", ascending=False)
model_info[["R^2", "MAE", "RMSE"]] = model_info[["R^2", "MAE", "RMSE"]].astype(float)
print(model_info)

# Додамо візуалізацію по показнику R^2
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='R^2', y='Model', data=model_info, hue='Model',
        palette='viridis', legend=False)

# підписи значень
for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.xlim(0.3, 0.9)
plt.title('Показники R^2 по моделям')
plt.xlabel('R^2')
plt.ylabel('Моделі')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "best_model_R^2.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'best_model_R^2.png'}")

# Додамо візуалізацію по показнику RMSE
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='RMSE', y='Model', data=model_info, hue='Model',
        palette='viridis', legend=False)

# підписи значень
for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.xlim(400, 1200)
plt.title('Показники RMSE по моделям')
plt.xlabel('RMSE')
plt.ylabel('Моделі')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "best_model_RMSE.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'best_model_RMSE.png'}")


# # Додамо нові дві колонки: Медіанна ціна по категорії і Медіанна ціна по дизайнеру і замінимо в ознаках
# колонку 'category' і 'designer'.

median_price = df.groupby(['category'])['price'].median()
median_dsgn =  df.groupby(['designer'])['price'].median()

df = df.set_index(['category'])
df['category_median_price'] = median_price
df = df.reset_index()

df = df.set_index(['designer'])
df['designer_median_price'] = median_dsgn
df = df.reset_index()
print(df.head())
print(df.info())

X1 = df[['category_median_price', 'designer_median_price', 'other_colors', 'width', 'height', 'depth']]
y1 = df['price']

# Ділимо на числові і категоріальні ознаки. Так як 'category_median_price', 'designer_median_price' зараз числа,# то вони потрапляють в числові ознаки
num_cols = ['width', 'height', 'depth', 'category_median_price', 'designer_median_price']
cat_cols = ['other_colors']
# Пайплайн для числових:
# 1) заповнюємо пропуски медіаною, це зроблено попередньо
# 2) масштабуємо StandardScaler (важливо для KNN)
numeric_pipeline = Pipeline(
    steps=[
        # ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
# Пайплайн для категоріальних:
# 1) заповнюємо пропуски модою (most_frequent) або можна "Unknown", все заповнено
# 2) кодуємо OneHotEncoder (бо порядку немає)
categorical_pipeline = Pipeline(
    steps=[
        # ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# Об’єднуємо все в один препроцесор
preprocessor_new = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ]
)
# Пробуємо тепер вибрати кращу модель
def getBestRegressor_new(X1, y1):
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    models = [
        LinearRegression(),
        LassoCV(),
        RidgeCV(),
        SVR(kernel="linear"),
        KNeighborsRegressor(n_neighbors=16),
        DecisionTreeRegressor(max_depth=10, random_state=42),
        RandomForestRegressor(random_state=42),
        GradientBoostingRegressor(),
        BaggingRegressor(random_state=42),
        XGBRegressor(objective="reg:squarederror", random_state=42)
    ]

    TestModels_new = pd.DataFrame()
    res = {}
    tmp = {}

    for model in models:
        model_pipeline = Pipeline(steps=[
            ("prepr", preprocessor_new),
            ("model", model)
        ])

        m = str(model)
        tmp["Model"] = m[: m.index("(")]
        model_pipeline.fit(X1_train, y1_train)
        tmp["R^2"] = "{:.5f}".format(model_pipeline.score(X1_test, y1_test))
        tmp["MAE"] = "{:.5f}".format(
            mean_absolute_error(model_pipeline.predict(X1_test), y1_test)
        )
        tmp["RMSE"] = "{:.5f}".format(
            np.sqrt(
                mean_squared_error(model_pipeline.predict(X1_test), y1_test)
            )
        )

        TestModels_new = pd.concat([TestModels_new, pd.DataFrame([tmp])])
    TestModels_new.set_index("Model", inplace=True)
    res["model"] = TestModels_new
    res["X1_train"] = X1_train
    res["y1_train"] = y1_train
    res["X1_test"] = X1_test
    res["y1_test"] = y1_test
    return res
# Виклик функції
model = getBestRegressor_new(X1, y1)
model_info = model["model"].sort_values(by="R^2", ascending=False)
model_info[["R^2", "MAE", "RMSE"]] = model_info[["R^2", "MAE", "RMSE"]].astype(float)
print(f"\nПісля того як зроблено ще трішки чистки: \n{model_info}")

# Додамо візуалізацію по показнику R^2
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='R^2', y='Model', data=model_info, hue='Model',
        palette='viridis', legend=False)

# підписи значень
for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.xlim(0.3, 0.9)
plt.title('Показники R^2_new по моделям')
plt.xlabel('R^2_new')
plt.ylabel('Моделі')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "best_model_R^2_new.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'best_model_R^2_new.png'}")

# Додамо візуалізацію по показнику RMSE
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='RMSE', y='Model', data=model_info, hue='Model',
        palette='viridis', legend=False)

# підписи значень
for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.xlim(400, 1200)
plt.title('Показники RMSE_NEW по моделям')
plt.xlabel('RMSE_NEW')
plt.ylabel('Моделі')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "best_model_RMSE_NEW.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'best_model_RMSE_NEW.png'}")


# Висновок. Після того як я створив нові колонки 'category_median_price', 'designer_median_price' заповнені
# середніми значеннями, а колонки 'category' і 'designer' виключив, то показники стали трішечки гірші
#               До
#                                R^2        MAE     RMSE
# XGBRegressor               0.81565  359.71956   589.25760
# RandomForestRegressor      0.81515  349.98842   590.05761
# BaggingRegressor           0.80068  359.70730   612.71673
#
#              після
#                                R^2        MAE     RMSE
# XGBRegressor               0.80721  356.77815  602.59354
# RandomForestRegressor      0.80653  356.32085  603.66029
# BaggingRegressor           0.78844  370.38438  631.25364

# # GridSearchCV.  Підбираємо найкращі параметри для моделі XGBRegressor
# 1 частина для XGBRegressor

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
# Повний pipeline
xgb_new = Pipeline([
    ('prepr', preprocessor_new),
    ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
])
# Параметри
param_grid = {
    'model__max_depth': [4, 6, 10, 30, 50, 75],
    'model__n_estimators': [50, 75, 85, 95, 125, 175],
    'model__colsample_bytree': [0.2, 0.6, 0.8],
    'model__min_child_weight': [5, 7, 10],
    'model__learning_rate': [0.01, 0.1, 0.3]
}
grid_search = GridSearchCV(
    estimator=xgb_new,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X1_train, y1_train)

print("Best Estimator :", grid_search.best_estimator_)
print("Best Score     :", grid_search.best_score_)
print("\nНа тестовій вибірці:")
print('R^2            : {:.5f}'.format(r2_score(y1_test, grid_search.predict(X1_test))))
print('MAE            : {:.5f}'.format(mean_absolute_error(grid_search.predict(X1_test), y1_test)))
print('RMSE           : {:.5f}'.format(np.sqrt(mean_squared_error(grid_search.predict(X1_test), y1_test))))

# Дослідження впливу факторів моделі даних
best_model = grid_search.best_estimator_
model = best_model.named_steps['model']
feature_importances = model.feature_importances_
numeric_features = ['depth', 'height', 'width', 'category_median_price', 'designer_median_price']
categorical_features = ['other_colors']

# Отримання назв колонок після препроцесингу
cat_columns = best_model.named_steps['prepr'].named_transformers_['cat'].named_steps[
    'onehot'].get_feature_names_out(categorical_features)
final_columns = np.concatenate([numeric_features, cat_columns])

# Створення DataFrame для важливості факторів
importance_df = pd.DataFrame({
    'Feature': final_columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Групування важливості по основним факторам
grouped_importances = {key: 0 for key in ['category_median_price', 'designer_median_price', 'other_colors', 'depth', 'height', 'width']}
for _, row in importance_df.iterrows():
    for key in grouped_importances:
        if row['Feature'].startswith(key):
            grouped_importances[key] += row['Importance']
            break

grouped_importances_df = pd.DataFrame.from_dict(grouped_importances, orient='index', columns=['Importance'])
grouped_importances_df = grouped_importances_df.sort_values(by='Importance', ascending=False)
grouped_importances_df['Importance (%)'] = (grouped_importances_df['Importance'] / grouped_importances_df[
    'Importance'].sum()) * 100
print(f"\nВплив ознак:\n{grouped_importances_df}")

# Візуалізація впливу ознак
plt.figure(figsize=(12,8))
ax = sns.barplot( x='Importance', y=grouped_importances_df.index, data=grouped_importances_df,
                  hue = grouped_importances_df.index,palette='viridis', legend=False)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f")
plt.title('Вплив ознак')
plt.xlabel('Важливість')
plt.ylabel('Ознаки')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "influence_of_traits.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'influence_of_traits.png'}")

# Висновок. Було                  R^2        MAE     RMSE
# # XGBRegressor               0.80721  356.77815  602.59354

# Покращилися трішки всі параметри
# Стало                        0.81138  349.18418  596.04932
# Best Estimator:
#                               feature_types=None, feature_weights=None,
#                               gamma=None, grow_policy=None,
#                               importance_type=None,
#                               interaction_constraints=None, learning_rate=0.1,
#                               max_bin=None, max_cat_threshold=None,
#                               max_cat_to_onehot=None, max_delta_step=None,
#                               max_depth=50, max_leaves=None, min_child_weight=5,
#                               missing=nan, monotone_constraints=None,
#                               multi_strategy=None, n_estimators=85, n_jobs=None,
#                               num_parallel_tree=None, ...))])
#
# Best Score     : 0.8027776892559139
#
# На тестовій вибірці:
# R^2            : 0.81138
# MAE            : 349.18418
# RMSE           : 596.04932

# Вплив ознак найбільше це глибина і ширина
#                         Importance     Importance (%)
# depth                    0.294041       29.404078
# width                    0.206342       20.634209
# category_median_price    0.144621       14.462120
# designer_median_price    0.141917       14.191723
# other_colors             0.125414       12.541395
# height                   0.087665        8.766474

#  **************************************************
# Кросвалідація
parameters = {
        'model__max_depth':[50],
        'model__n_estimators':[85],
        'model__min_child_weight':[5],
        'model__colsample_bytree':[0.6],
        'model__learning_rate':[0.1]}

grid_search = GridSearchCV(xgb_new, parameters, cv=5, n_jobs=-1)
grid_search.fit(X1_train, y1_train)
scores = cross_val_score(grid_search, X1_train, y1_train, cv=5, scoring='r2')
scores_RMSE = cross_val_score(grid_search, X1_train, y1_train, cv=5, scoring='neg_mean_squared_error')

# Приводимо показник RMSE до коректного вигляду
scores_RMSE = np.sqrt(-scores_RMSE)

print(f"\nПоказники R^2  на кожному фолді: {scores}")
print(f"Середній показник R^2 : {scores.mean()}")
print(f"\nПоказники RMSE на кожному фолді: {scores_RMSE}")
print(f"Середній показник RMSE: {scores_RMSE.mean()}")

# Візуалізація по кросвалідації
folds = np.arange(1, len(scores) + 1)
cv_df = pd.DataFrame({
    'Fold': folds,
    'R^2': scores,
    'RMSE': scores_RMSE
})

# Візуалізація для R^2
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Fold', y='R^2', data=cv_df, hue='R^2', palette='viridis', legend=False )
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f")

plt.title('R^2 по фолдах')
plt.savefig(PLOTS_DIR / "cross_val_R^2.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'cross_val_R^2.png'}")

# Візуалізація для RMSE
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Fold', y='RMSE', hue='RMSE', data=cv_df, palette='viridis', legend=False)

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

plt.title('RMSE по фолдах')
plt.savefig(PLOTS_DIR / "cross_val_RMSE.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"Saved: {PLOTS_DIR / 'cross_val_RMSE.png'}")

# Висновок. Після Кросвалідації трішечки показники погіршилися. На GridSearchCV було R^2: 0.81138, RMSE: 596.04932
# Показники R^2  на кожному фолді: [0.80307469 0.83510639 0.80711674 0.7256417  0.84294892]
# Середній показник R^2 к: 0.8027776892559139
# Показники RMSE на кожному фолді: [640.67823142 590.3645898  569.09266936 636.30733902 613.27739421]
# Середній показник RMSE: 609.944044762822


# *****************************************************************************************
# *****************************************************************************************
# Глобальний вплив має колонка 'name'
#                           Importance      Importance (%)
# # name                     0.854770       85.477041
# Додатково було добавлено в ознаки колонку 'name', але після кросвалідації показники погіршилися
# Середній показник R^2 : 0.8116380657390831 був 0.83459 GridSearchCV і після першої чистки 0.82152, а сама перше 0.84228
# Середній показник RMSE: 595.5737348768591 був 558.17145 GridSearchCV і після першої чистки 579.79852, , а сама перше 545.04285
# але вирішив це не відправляти, залишити попередній варіант, добавив трішки візуалів де не вистачило і вплив ознак

# Було добавлено колонку name в категоріальні ознаки. Дані були такі
#                               R^2        MAE        RMSE
# Model
# XGBRegressor               0.84228  343.25469   545.04285
# RandomForestRegressor      0.83101  332.78178   564.16931
# BaggingRegressor           0.81458  343.59090   590.96955
# GradientBoostingRegressor  0.75481  455.88315   679.57677
# RidgeCV                    0.73128  483.08395   711.43395
# LinearRegression           0.72473  490.32648   720.04943
# KNeighborsRegressor        0.71967  456.62341   726.64246
# DecisionTreeRegressor      0.71469  462.45142   733.06977
# LassoCV                    0.69544  509.95156   757.38495
# SVR                        0.38414  602.97861  1077.01744

# Після того як приведені колонки медіана по дизайнерам і категоріям
# Після того як зроблено ще трішки чистки:
#                                R^2        MAE       RMSE
# Model
# XGBRegressor               0.82152  348.30479  579.79852
# RandomForestRegressor      0.80878  344.17151  600.13579
# BaggingRegressor           0.78110  374.06972  642.10925
# GradientBoostingRegressor  0.76307  435.50917  668.02935
# RidgeCV                    0.74357  457.54756  694.96721
# LinearRegression           0.73769  461.64187  702.88986
# LassoCV                    0.71647  491.61551  730.77559
# KNeighborsRegressor        0.70578  459.09251  744.41973
# DecisionTreeRegressor      0.68521  470.33747  770.01024
# SVR                        0.51307  564.87647  957.66865

#
# Best Estimator :feature_types=None, feature_weights=None,
# #                               gamma=None, grow_policy=None,
# #                               importance_type=None,
# #                               interaction_constraints=None, learning_rate=0.1,
# #                               max_bin=None, max_cat_threshold=None,
# #                               max_cat_to_onehot=None, max_delta_step=None,
# #                               max_depth=50, max_leaves=None, min_child_weight=5,
# #                               missing=nan, monotone_constraints=None,
# #                               multi_strategy=None, n_estimators=95, n_jobs=None,
# #                               num_parallel_tree=None, ...))])
#
# Best Score     : 0.8116380657390831
#
# На тестовій вибірці:
# R^2            : 0.83459
# MAE            : 326.55519
# RMSE           : 558.17145
#
# Вплив ознак: Глобальний вплив має name
#                        Importance  Importance (%)
# name                     0.854770       85.477041
# depth                    0.044225        4.422471
# width                    0.023797        2.379685
# designer_median_price    0.023103        2.310277
# category_median_price    0.023097        2.309739
# other_colors             0.019291        1.929099
# height                   0.011717        1.171689
# Saved: plots\influence_of_traits.png

# Показники R^2  на кожному фолді: [0.82131908 0.83868149 0.82953374 0.72464074 0.84401528]
# Середній показник R^2 : 0.8116380657390831
#
# Показники RMSE на кожному фолді: [610.27881579 583.9295952  535.0014312  637.46701583 611.19181637]
# Середній показник RMSE: 595.5737348768591
# Saved: plots\cross_val_R^2.png
# Saved: plots\cross_val_RMSE.png


# Висновок. Після Кросвалідації трішечки показники погіршилися. Середній показник R^2 : 0.8116380657390831 був 0.83459
# Середній показник RMSE: 595.5737348768591 був 558.17145