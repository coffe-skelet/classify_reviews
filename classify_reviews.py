#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для автоматической классификации текстовых отзывов.

Поддерживает два режима работы:
    1. ML (scikit-learn): если в CSV есть колонка 'true_label' с достаточным
       количеством примеров на каждый класс.
    2. Keyword-based: если разметки нет или её недостаточно.

Зависимости:
    pip install pandas scikit-learn numpy

Пример входного CSV:
    review_text,true_label
    "Отличный товар, всё работает!","позитивный"
    "Ужасное качество, не рекомендую","негативный"
    "Скажите, а есть ли доставка?","нейтральный"
"""

import os
import re
import sys
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ==================== КОНФИГУРАЦИЯ ====================

# Путь к входному CSV-файлу
INPUT_CSV_PATH = "reviews.csv"

# Путь к выходному CSV-файлу (если None, будет создан автоматически)
OUTPUT_CSV_PATH = None  # Пример: "reviews_classified.csv"

# Названия колонок
TEXT_COLUMN = "review_text"
LABEL_COLUMN = "true_label"
PREDICTED_COLUMN = "predicted_label"

# Минимальное количество примеров на класс для обучения ML-модели
MIN_SAMPLES_PER_CLASS = 5

# Доля тестовой выборки для оценки ML-модели (0.0 - 1.0)
TEST_SIZE = 0.2

# Случайное зерно для воспроизводимости
RANDOM_STATE = 42

# ==================== СЛОВАРИ KEYWORD-ПОДХОДА ====================

POSITIVE_WORDS = {
    # Русский
    "хороший", "отличный", "супер", "класс", "классный", "рекомендую",
    "люблю", "быстро", "качественно", "понравилось", "доволен", "довольна",
    "спасибо", "идеально", "прекрасно", "замечательно", "отлично", "великолепно",
    "превосходно", "рад", "рада", "счастлив", "удобно", "просто", "надежно",
    "красиво", "впечатляет", "восхищен", "советую", "топ", "лучший", "шикарно",
    "благодарен", "порадовал", "удобный", "качественный", "надежный", "быстрый",
    "хорошо", "здорово", "круто", "прикольно", "умница", "молодцы", "восхитительно",
    "безупречно", "комфортно", "эффективно", "полезно", "интересно", "увлекательно",
    # English
    "good", "great", "excellent", "amazing", "awesome", "perfect", "love",
    "best", "fantastic", "wonderful", "happy", "satisfied", "recommend",
    "nice", "beautiful", "easy", "fast", "quick", "reliable", "thanks",
    "thank", "brilliant", "outstanding", "superb", "lovely", "impressive",
    "pleasant", "smooth", "flawless", "decent", "solid", "helpful"
}

NEGATIVE_WORDS = {
    # Русский
    "плохой", "ужасный", "говно", "не работает", "сломался", "сломалась",
    "долго", "плохо", "разочарован", "разочарована", "ненавижу", "проблема",
    "ошибка", "жалоба", "ужасно", "отвратительно", "не рекомендую",
    "не советую", "грустно", "обидно", "развод", "обман",
    "дешевка", "хлам", "брак", "глючит", "тормозит", "неудобно", "жуть",
    "ужас", "отстой", "позор", "недоволен", "недовольна", "плохое",
    "не работал", "не работала", "не смог", "не смогла", "не удалось",
    "отвратительный", "плохое качество", "не покупайте", "не заказывайте",
    "ужасное", "отвратительное", "глючный", "бракованный", "неполадка",
    "возврат", "ненависть", "испорчено", "дефект", "кривой", "косой",
    # English
    "bad", "terrible", "awful", "worst", "hate", "disappointed", "poor",
    "horrible", "disgusting", "broken", "defective", "slow", "useless",
    "waste", "problem", "issue", "error", "fail", "failed", "sucks", "crap",
    "junk", "never again", "not recommend", "unhappy", "angry", "difficult",
    "hard", "impossible", "ridiculous", "shame", "scam", "fraud", "worst",
    "unfortunately", "regret", "mess", "nightmare", "disaster", "trash"
}

NEUTRAL_INDICATORS = {
    # Русский
    "вопрос", "как", "что", "где", "когда", "почему", "зачем", "какой",
    "сколько", "можно ли", "есть ли", "подскажите", "объясните", "уточните",
    "интересно", "хочу узнать", "не понял", "не поняла", "не ясно",
    "уточнение", "спрашиваю", "информация", "узнать", "поясните",
    # English
    "question", "how", "what", "where", "when", "why", "which", "whether",
    "ask", "wonder", "curious", "confused", "clarify", "explain", "information"
}


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def print_banner(text: str) -> None:
    """Выводит декоративный заголовок."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Загружает CSV-файл с обработкой ошибок кодировки.
    
    Args:
        filepath: Путь к CSV-файлу
        
    Returns:
        DataFrame с данными
        
    Raises:
        SystemExit: при критических ошибках
    """
    if not os.path.exists(filepath):
        print(f"\n[ОШИБКА] Файл не найден: {os.path.abspath(filepath)}")
        print(f"[INFO] Текущая рабочая директория: {os.getcwd()}")
        print("[INFO] Создайте файл или измените переменную INPUT_CSV_PATH в начале скрипта.")
        sys.exit(1)

    # Пробуем UTF-8
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        print(f"\n[OK] Файл успешно загружен в кодировке UTF-8: {filepath}")
    except UnicodeDecodeError:
        print("\n[WARNING] Не удалось прочитать файл в UTF-8. Пробуем cp1251...")
        try:
            df = pd.read_csv(filepath, encoding="cp1251")
            print(f"[OK] Файл загружен в кодировке cp1251: {filepath}")
        except Exception as e:
            print(f"[ОШИБКА] Не удалось прочитать файл: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ОШИБКА] При загрузке файла: {e}")
        sys.exit(1)

    # Проверка обязательной колонки
    if TEXT_COLUMN not in df.columns:
        print(f"\n[ОШИБКА] В файле отсутствует обязательная колонка '{TEXT_COLUMN}'")
        print(f"[INFO] Доступные колонки: {list(df.columns)}")
        sys.exit(1)

    # Очистка пустых строк
    initial_len = len(df)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).replace("nan", "")
    df = df[df[TEXT_COLUMN].str.strip() != ""]
    dropped = initial_len - len(df)
    
    if dropped > 0:
        print(f"[INFO] Удалено пустых/некорректных строк: {dropped}")
    
    if len(df) == 0:
        print("[ОШИБКА] После очистки не осталось данных для обработки.")
        sys.exit(1)

    print(f"[INFO] Всего строк к обработке: {len(df)}")
    return df.reset_index(drop=True)


def preprocess_text(text: str) -> str:
    """
    Базовая предобработка текста для ML-модели.
    
    Args:
        text: Исходный текст отзыва
        
    Returns:
        Очищенный текст в нижнем регистре
    """
    if not isinstance(text, str):
        return ""
    # Приводим к нижнему регистру
    text = text.lower()
    # Удаляем лишние пробелы
    text = " ".join(text.split())
    return text


def tokenize_simple(text: str) -> List[str]:
    """
    Простая токенизация для keyword-based подхода.
    Извлекает слова, состоящие из букв (поддержка русского и английского).
    """
    if not text:
        return []
    # Оставляем только буквы и пробелы
    cleaned = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\s]", " ", text.lower())
    return cleaned.split()


# ==================== КЛАССИФИКАТОРЫ ====================

class KeywordClassifier:
    """
    Классификатор на основе словарей ключевых слов.
    Определяет тональность по балансу позитивных/негативных/нейтральных слов.
    """
    
    def __init__(self):
        self.positive = POSITIVE_WORDS
        self.negative = NEGATIVE_WORDS
        self.neutral = NEUTRAL_INDICATORS
        
    def predict(self, texts: List[str]) -> List[str]:
        """
        Классифицирует список текстов.
        
        Args:
            texts: Список исходных текстов
            
        Returns:
            Список меток классов
        """
        return [self._classify_single(t) for t in texts]
    
    def _classify_single(self, text: str) -> str:
        """Классифицирует один текст."""
        if not text or not isinstance(text, str):
            return "нейтральный"
            
        text_lower = text.lower()
        tokens = tokenize_simple(text)
        if not tokens:
            return "нейтральный"
            
        # Считаем многословные фразы (весомее, чем отдельные слова)
        pos_count = sum(2 for w in self.positive if " " in w and w in text_lower)
        neg_count = sum(2 for w in self.negative if " " in w and w in text_lower)
        neut_count = sum(1 for w in self.neutral if " " in w and w in text_lower)
        
        # Считаем одиночные слова по токенам
        pos_count += sum(1 for t in tokens if t in self.positive)
        neg_count += sum(1 for t in tokens if t in self.negative)
        neut_count += sum(1 for t in tokens if t in self.neutral)
        
        # Принятие решения
        if neg_count > pos_count:
            return "негативный"
        elif pos_count > neg_count:
            return "позитивный"
        elif neut_count > 0:
            return "нейтральный"
        else:
            return "нейтральный"


class MLClassifier:
    """
    ML-классификатор на основе TF-IDF и LogisticRegression.
    """
    
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=preprocess_text,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                C=10.0
            ))
        ])
        self.is_trained = False
        self.classes_ = None
        
    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        Обучает модель на размеченных данных.
        """
        print("[INFO] Обучение ML-модели (TF-IDF + LogisticRegression)...")
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        self.classes_ = list(self.pipeline.classes_)
        print(f"[OK] Модель обучена. Классы: {self.classes_}")
        
    def predict(self, texts: List[str]) -> List[str]:
        """
        Предсказывает метки для текстов.
        """
        if not self.is_trained:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit().")
        return self.pipeline.predict(texts).tolist()
    
    def evaluate(self, texts: List[str], labels: List[str]) -> dict:
        """
        Оценивает качество на тестовой выборке.
        
        Returns:
            Словарь с метриками и предсказаниями
        """
        preds = self.pipeline.predict(texts)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, zero_division=0)
        cm = confusion_matrix(labels, preds, labels=self.classes_)
        
        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": preds
        }


# ==================== ОЦЕНКА И СТАТИСТИКА ====================

def print_statistics(df: pd.DataFrame, pred_col: str, true_col: Optional[str] = None) -> None:
    """
    Выводит статистику по распределению классов.
    
    Args:
        df: DataFrame с предсказаниями
        pred_col: Название колонки с предсказаниями
        true_col: Название колонки с истинными метками (опционально)
    """
    print_banner("РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАННЫХ КЛАССОВ")
    
    counts = df[pred_col].value_counts().sort_index()
    total = len(df)
    
    print(f"{'Класс':<20} {'Количество':>10} {'Процент':>10}")
    print("-" * 45)
    for cls, cnt in counts.items():
        pct = (cnt / total) * 100
        print(f"{cls:<20} {cnt:>10} {pct:>9.1f}%")
    print("-" * 45)
    print(f"{'ИТОГО':<20} {total:>10} {100.0:>9.1f}%")
    
    if true_col and true_col in df.columns:
        print_banner("СРАВНЕНИЕ С ИСТИННЫМИ МЕТКАМИ")
        
        # Точность
        accuracy = accuracy_score(df[true_col], df[pred_col])
        print(f"Accuracy (общая точность): {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(df[true_col], df[pred_col], zero_division=0))
        
        # Confusion Matrix
        labels = sorted(df[true_col].unique())
        cm = confusion_matrix(df[true_col], df[pred_col], labels=labels)
        
        print("Confusion Matrix:")
        print(f"{'':>12}", end="")
        for lbl in labels:
            print(f"{lbl:>12}", end="")
        print()
        
        for i, true_lbl in enumerate(labels):
            print(f"{true_lbl:>12}", end="")
            for j in range(len(labels)):
                print(f"{cm[i, j]:>12}", end="")
            print()
        print("\n(Строки — истинные метки, столбцы — предсказанные)")


def save_results(df: pd.DataFrame, output_path: Optional[str], input_path: str) -> str:
    """
    Сохраняет результаты в CSV-файл.
    
    Returns:
        Путь к сохраненному файлу
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_classified{ext}"
    
    try:
        # utf-8-sig добавляет BOM, чтобы Excel корректно открывал русский текст
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] Результаты сохранены: {os.path.abspath(output_path)}")
        return output_path
    except Exception as e:
        print(f"\n[ОШИБКА] Не удалось сохранить файл: {e}")
        sys.exit(1)


# ==================== ОСНОВНАЯ ЛОГИКА ====================

def main() -> None:
    """Главная функция скрипта."""
    print_banner("КЛАССИФИКАЦИЯ ТЕКСТОВЫХ ОТЗЫВОВ")
    print(f"Входной файл:  {INPUT_CSV_PATH}")
    print(f"Выходной файл: {OUTPUT_CSV_PATH or 'AUTO'}")
    
    # 1. Загрузка данных
    df = load_data(INPUT_CSV_PATH)
    
    # 2. Определение стратегии
    has_labels = LABEL_COLUMN in df.columns
    
    if has_labels:
        # Очищаем метки от NaN и пустых строк
        df[LABEL_COLUMN] = df[LABEL_COLUMN].replace(["nan", ""], np.nan)
        df_clean = df.dropna(subset=[LABEL_COLUMN]).copy()
        
        label_counts = df_clean[LABEL_COLUMN].value_counts()
        valid_classes = label_counts[label_counts >= MIN_SAMPLES_PER_CLASS]
        
        if len(valid_classes) >= 2:
            use_ml = True
            df_ml = df_clean[df_clean[LABEL_COLUMN].isin(valid_classes.index)].copy()
            print(f"\n[INFO] Найдена разметка. Пригодных классов: {len(valid_classes)}")
            for cls, cnt in valid_classes.items():
                print(f"       - '{cls}': {cnt} примеров")
        else:
            use_ml = False
            print(f"\n[WARNING] Недостаточно данных для ML (нужно ≥{MIN_SAMPLES_PER_CLASS} примеров на класс).")
            print("[INFO] Используем keyword-based классификатор.")
            df_ml = df.copy()
    else:
        use_ml = False
        print(f"\n[INFO] Колонка '{LABEL_COLUMN}' не найдена. Используем keyword-based классификатор.")
        df_ml = df.copy()
    
    # 3. Классификация
    texts = df[TEXT_COLUMN].astype(str).tolist()
    
    if use_ml:
        classifier = MLClassifier()
        texts_ml = df_ml[TEXT_COLUMN].astype(str).tolist()
        labels_ml = df_ml[LABEL_COLUMN].astype(str).tolist()
        
        # Если данных достаточно, делаем оценку на тестовой выборке
        if len(df_ml) >= 20 and TEST_SIZE > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                texts_ml, labels_ml, test_size=TEST_SIZE, 
                random_state=RANDOM_STATE, stratify=labels_ml
            )
            classifier.fit(X_train, y_train)
            
            # Оценка
            print_banner("ОЦЕНКА ML-МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
            metrics = classifier.evaluate(X_test, y_test)
            print(f"Accuracy на тесте: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"\nClassification Report (test):")
            print(metrics["classification_report"])
            
            # Переобучаем на всех данных для финального предсказания
            print("[INFO] Переобучение модели на всех размеченных данных...")
            classifier.fit(texts_ml, labels_ml)
        else:
            # Мало данных — обучаем сразу на всех
            classifier.fit(texts_ml, labels_ml)
        
        # Предсказание для ВСЕХ строк (включая те, что не вошли в ML-выборку)
        predictions = classifier.predict(texts)
        
    else:
        classifier = KeywordClassifier()
        predictions = classifier.predict(texts)
    
    # 4. Сохранение результатов
    df[PREDICTED_COLUMN] = predictions
    
    # 5. Вывод статистики
    print_statistics(df, PREDICTED_COLUMN, LABEL_COLUMN if has_labels else None)
    
    # 6. Сохранение в файл
    save_results(df, OUTPUT_CSV_PATH, INPUT_CSV_PATH)
    
    print_banner("ГОТОВО")
    print("Спасибо за использование скрипта!")


if __name__ == "__main__":
    main()
