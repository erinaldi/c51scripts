# Import smtplib for the actual sending function
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from email.header    import Header
from email.mime.text import MIMEText
from smtplib         import SMTP_SSL

login, password = 'pyo0220@gmail.com', 'wxngkiujxlmyfxey' 
recipients = ['walksloud@gmail.com']

# create message
msg = MIMEText('Andre, start writing papers!', 'plain', 'utf-8')
msg['Subject'] = Header('Friendly daily reminder for Andre Walker-Loud', 'utf-8')
msg['From'] = login
msg['To'] = ", ".join(recipients)

# send it via gmail
s = SMTP_SSL('smtp.gmail.com', 465, timeout=10)
s.set_debuglevel(1)
try:
    s.login(login, password)
    s.sendmail(msg['From'], recipients, msg.as_string())
finally:
    s.quit()
