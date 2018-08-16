
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import smtplib

def send_mail_message(subject, message, tag = None):

    if tag is None:
        tag_str = ""
    else:
        tag_str = "+" + tag


    # create message object instance
    msg = MIMEMultipart()

    # setup the parameters of the message
    password = "BeepBoop01"
    msg['From'] = "hannesprogram@gmail.com"
    msg['To'] = "hannes.von.essen" + tag_str + "@gmail.com"
    msg['Subject'] = subject

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    # create server
    server = smtplib.SMTP('smtp.gmail.com: 587')

    server.starttls()

    # Login Credentials for sending the mail
    server.login(msg['From'], password)

    # send the message via the server.
    server.sendmail(msg['From'], msg['To'], msg.as_string())

    server.quit()

    print "successfully sent email to %s" % (msg['To'])

def send_mail_message_with_image(subject, message, image, tag = None):

    if tag is None:
        tag_str = ""
    else:
        tag_str = "+" + tag


    # create message object instance
    msg = MIMEMultipart()

    # setup the parameters of the message
    password = "BeepBoop01"
    msg['From'] = "hannesprogram@gmail.com"
    msg['To'] = "hannes.von.essen" + tag_str + "@gmail.com"
    msg['Subject'] = subject

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    try:
        fp = open(image, 'rb')
        msgImage = MIMEImage(fp.read())
        fp.close()
        msg.attach(msgImage)
    except IOError:
        print "Couldn't include image '" + image + "'."
        msg.attach(MIMEText("\n\nCouldn't include image '" + image + "'.", 'plain'))

    # create server
    server = smtplib.SMTP('smtp.gmail.com: 587')

    server.starttls()

    # Login Credentials for sending the mail
    server.login(msg['From'], password)

    # send the message via the server.
    server.sendmail(msg['From'], msg['To'], msg.as_string())

    server.quit()

    print "successfully sent email to %s" % (msg['To'])



if __name__ == "__main__":
    send_mail_message_with_image("HEJ", "message with image", "../andungarna.jpg")