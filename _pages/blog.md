---
layout: archive
title: "Blog"
permalink: /blog/
author_profile: true
---
{% include group-by-array collection=site.blog field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for blog in blogs %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
