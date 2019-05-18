def style():
    return """
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css"
    integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">"""

def div_boks_alert(text):
    html_text = """<div class="alert alert-danger" role="alert">
    {0}
    </div>""".format(text)
    return html_text

def div_boks_success(text):
    html_text = """<div class="alert alert-primary" role="alert">
      {0}
    </div>""".format(text)
    return html_text

def div_card(header, title, text):
    html_text = """<div class="card bg-light mb-3" style="max-width: 30%;">
  <div class="card-header">{0}</div>
  <div class="card-body">
    <h5 class="card-title">{1}</h5>
    <p class="card-text">{2}</p>
  </div>
</div>""".format(header,title, text)
    return html_text

def div_card_group(text):
    return """<div class="card-group">{0}</div>""".format(text)

def create_main_page(body):
    return """
<html>
    <head>
{0}
        <style>
        body {margin:0 100; background:whitesmoke;
        
        counter-reset: h2counter;
    }
    h1 {
        counter-reset: h2counter;
    }
    h2:before {
        content: counter(h2counter) ".\0000a0\0000a0";
        counter-increment: h2counter;
        counter-reset: h3counter;
    }
    h3:before {
        content: counter(h2counter) "." counter(h3counter) ".\0000a0\0000a0";
        counter-increment: h3counter;
    }
         </style>
    </head>
    <body>
    
        {1}
    </body>
</html>
    """.format(style(), body)
