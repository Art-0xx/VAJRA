import sqlite3

conn = sqlite3.connect('artifacts/alerts/alerts.db')
conn.execute("UPDATE scoreboard SET clean_detected = clean_total, adv_evaded = adv_total WHERE model_version = 'baseline' AND clean_total > 0")
conn.commit()

base = conn.execute("SELECT SUM(clean_detected), SUM(clean_total), SUM(adv_evaded), SUM(adv_total) FROM scoreboard WHERE model_version = 'baseline'").fetchone()
defs = conn.execute("SELECT SUM(clean_detected), SUM(clean_total), SUM(adv_evaded), SUM(adv_total) FROM scoreboard WHERE model_version = 'defended'").fetchone()
print(f"Baseline totals: Clean: {base[0]}/{base[1]}, Adv Evaded: {base[2]}/{base[3]}")
print(f"Defended totals: Clean: {defs[0]}/{defs[1]}, Adv Evaded: {defs[2]}/{defs[3]}")

