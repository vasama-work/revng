void f() {
	int i;
	if (condition) {
		i = get_value();
		set_value(i * i);
	}
}

void f() {
	if (condition) {
		int i;
		i = get_value();
		set_value(i * i);
	}
}
