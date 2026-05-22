int32_t f(struct_123 *p) {
	return *(int32_t*) ((generic64_t) p + 4);
}

int32_t f(struct_123 *p) {
	return p->offset_4;
}
