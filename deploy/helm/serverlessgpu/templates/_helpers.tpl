{{/* Standard helm naming */}}
{{- define "serverlessgpu.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "serverlessgpu.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "serverlessgpu.labels" -}}
app.kubernetes.io/name: {{ include "serverlessgpu.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "serverlessgpu.gateway.selectorLabels" -}}
app.kubernetes.io/name: {{ include "serverlessgpu.name" . }}
app.kubernetes.io/component: gateway
{{- end -}}

{{- define "serverlessgpu.redis.selectorLabels" -}}
app.kubernetes.io/name: {{ include "serverlessgpu.name" . }}
app.kubernetes.io/component: redis
{{- end -}}

{{- define "serverlessgpu.postgres.selectorLabels" -}}
app.kubernetes.io/name: {{ include "serverlessgpu.name" . }}
app.kubernetes.io/component: postgres
{{- end -}}

{{- define "serverlessgpu.postgres-exporter.selectorLabels" -}}
app.kubernetes.io/name: {{ include "serverlessgpu.name" . }}
app.kubernetes.io/component: postgres-exporter
{{- end -}}

{{- define "serverlessgpu.web.selectorLabels" -}}
app.kubernetes.io/name: {{ include "serverlessgpu.name" . }}
app.kubernetes.io/component: web
{{- end -}}
