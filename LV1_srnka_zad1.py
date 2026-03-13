def total_euro(sati, eura_po_satu):
    return sati * eura_po_satu

sati = float(input("Radnih sati: "))
eura_po_satu = float(input("euro/h: "))

ukupno = total_euro(sati, eura_po_satu)

print(f"Ukupno: {ukupno} eura")
