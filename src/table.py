import polars as pl
from src.config import CONSUMER_CARDS_FILE, PREDICTIONS_DIR

def export_full_details():
    # 1. Читаем предсказания
    # Принудительно приводим card_number к строке сразу при чтении, если это возможно, 
    # или cast'им после.
    try:
        preds = pl.read_csv(PREDICTIONS_DIR / "final_submission.csv")
    except Exception as e:
        print(f"Ошибка чтения файла предсказаний: {e}")
        return

    # Фильтруем и приводим ID к строковому типу (Utf8)
    high_risk_list = (
        preds.filter(pl.col("score") >= 0.7)
        .select(pl.col("card_number").cast(pl.Utf8))
        .to_series()
        .to_list()
    )

    if not high_risk_list:
        print("Внимание: Карты с высоким риском (score >= 0.7) не найдены.")
        return

    print(f"Найдено {len(high_risk_list)} подозрительных карт. Читаю транзакции...")

    # 2. Читаем исходный файл консьюмеров и приводим его ID к строке
    cons_tx = pl.read_parquet(CONSUMER_CARDS_FILE).with_columns(
        pl.col("card_number").cast(pl.Utf8)
    )
    
    # 3. Фильтруем данные
    risk_details = cons_tx.filter(pl.col("card_number").is_in(high_risk_list))
    
    # 4. Группируем для аналитики
    # Убедимся, что сумма транзакций считается корректно
    mcc_report = (
        risk_details.group_by(["card_number", "mcc"])
        .agg(
            pl.len().alias("txn_count"), 
            pl.col("transaction_amount_kzt").sum().alias("total_spent")
        )
        .sort(["card_number", "txn_count"], descending=[False, True])
    )
    
    # 5. Сохраняем
    output_path = PREDICTIONS_DIR / "high_risk_mcc_details.csv"
    mcc_report.write_csv(output_path)
    print(f"Отчет успешно сохранен в: {output_path}")

if __name__ == "__main__":
    export_full_details()