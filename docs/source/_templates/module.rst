{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree:
{% for item in functions %}
   {{ item }}
{%- endfor %}

{% for item in functions %}
.. autofunction:: {{ item }}
{% endfor %}
{% endif %}
{% endblock %}
