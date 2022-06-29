from Deal import Deal


if __name__ == '__main__':
    
    watch_symbols = [1475]
    
    deal = Deal()
    deposit = deal.inquiry_deposit()
    print(f"預金残高：{int(deposit):,} 円")
    
