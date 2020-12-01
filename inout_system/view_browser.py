import flask
import pandas as pd
import os
from csv_process import read_csv, write_csv




html="""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="2; URL=">

    <title>DataFrame表示テスト</title>
</head>
<body>
{{table|safe}}
</body>
</html>
"""

app = flask.Flask(__name__)
@app.route('/')
def index():
    df = read_csv(today_csv)
    df.fillna("-", inplace=True)
    return flask.render_template_string(html, table=df.to_html(header='true'))

if __name__ == '__main__':
    now = pd.Timestamp.now()
    today_csv = now.strftime("%Y-%m-%d") + ".csv"
    base_csv = "data/arasi.csv"
    if os.path.exists(today_csv) == False:
        today_data = read_csv(base_csv)
        write_csv(today_data, today_csv)
    app.run(debug=True)