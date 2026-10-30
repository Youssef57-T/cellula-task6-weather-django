web: python manage.py collectstatic --noinput && gunicorn weather_site.wsgi --workers 2 --timeout 60 --bind 0.0.0.0:$PORT
