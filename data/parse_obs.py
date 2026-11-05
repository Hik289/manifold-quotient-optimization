import json
d=json.load(open("observability_results.json"))
print("raw gradient vertical ratio:", d["raw_gradient_vert_ratio"])
print()
hdr=("config","upd_vert_mean","upd_vert_max","bal_final","final_loss")
print("{:30s} {:>13s} {:>12s} {:>11s} {:>11s}".format(*hdr))
for k,v in d.items():
    if isinstance(v,dict):
        uv=v["update_vert"]
        print("{:30s} {:13.4f} {:12.4f} {:11.2e} {:11.2e}".format(
            k, uv["mean"], uv["max"], v["balance"]["final"], v["final_loss"]))
