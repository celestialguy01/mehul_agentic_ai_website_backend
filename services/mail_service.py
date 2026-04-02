from fastapi_mail import FastMail, MessageSchema
from services.mail_config import conf

async def send_contact_email(name: str, email: str, message: str):
    body = f"""
    New Contact Form Submission

    Name: {name}
    Email: {email}

    Message:
    {message}
    """

    message = MessageSchema(
        subject="New Contact Form Submission",
        recipients=[conf.MAIL_FROM],  # you receive it
        body=body,
        subtype="plain",
    )

    fm = FastMail(conf)
    await fm.send_message(message)